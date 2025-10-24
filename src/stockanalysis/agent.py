from typing import List

from langchain_core.messages import AIMessage

from stockanalysis.model import call_llm
from stockanalysis.prompts import (
    ACTION_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    VALIDATION_SYSTEM_PROMPT,
)
from stockanalysis.schemas import Answer, IsDone, Task, TaskList
from stockanalysis.tools import TOOLS
from stockanalysis.utils.logger import Logger
from stockanalysis.utils.ui import show_progress


class Agent:
    def __init__(self, max_steps: int = 20, max_steps_per_task: int = 5):
        self.logger = Logger()
        self.max_steps = max_steps            # global safety cap
        self.max_steps_per_task = max_steps_per_task

    # ---------- task planning ----------
    @show_progress("Planning tasks...", "Tasks planned")
    def plan_tasks(self, query: str) -> List[Task]:
        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in TOOLS])
        prompt = f"""
        Given the user query: "{query}",
        Create a list of tasks to be completed.
        Example: {{"tasks": [{{"id": 1, "description": "some task", "done": false}}]}}
        """
        system_prompt = PLANNING_SYSTEM_PROMPT.format(tools=tool_descriptions)
        try:
            response = call_llm(prompt, system_prompt=system_prompt, output_schema=TaskList)
            tasks = response.tasks
        except Exception as e:
            self.logger._log(f"Planning failed: {e}")
            tasks = [Task(id=1, description=query, done=False)]
        
        task_dicts = [task.dict() for task in tasks]
        self.logger.log_task_list(task_dicts)
        return tasks

    # ---------- ask LLM what to do ----------
    @show_progress("Thinking...", "")
    def ask_for_actions(self, task_desc: str, last_outputs: str = "") -> AIMessage:
        # last_outputs = textual feedback of what we just tried
        prompt = f"""
        We are working on: "{task_desc}".
        Here is a history of tool outputs from the session so far: {last_outputs}

        Based on the task and the outputs, what should be the next step?
        """
        try:
            return call_llm(prompt, system_prompt=ACTION_SYSTEM_PROMPT, tools=TOOLS)
        except Exception as e:
            self.logger._log(f"ask_for_actions failed: {e}")
            return AIMessage(content="Failed to get actions.")

    # ---------- ask LLM if task is done ----------
    @show_progress("Validating...", "")
    def ask_if_done(self, task_desc: str, recent_results: str) -> bool:
        prompt = f"""
        We were trying to complete the task: "{task_desc}".
        Here is a history of tool outputs from the session so far: {recent_results}

        Is the task done?
        """
        try:
            resp = call_llm(prompt, system_prompt=VALIDATION_SYSTEM_PROMPT, output_schema=IsDone)
            return resp.done
        except:
            return False

    # ---------- tool execution ----------
    def _execute_tool(self, tool, tool_name: str, inp_args):
        """Execute a tool with progress indication."""
        # Create a dynamic decorator with the tool name
        @show_progress(f"Executing {tool_name}...", "")
        def run_tool():
            return tool.run(inp_args)
        return run_tool()
    
    # ---------- confirm action ----------
    def confirm_action(self, tool: str, input_str: str) -> bool:
        # In production you'd ask the user; here we just log and auto-confirm
        # Risky tools are not implemented in this version.
        return True

    # ---------- main loop ----------
    def run(self, query: str):
        # Reset state
        step_count = 0
        last_actions = []
        session_outputs = []  # accumulate outputs for the whole session

        # Plan tasks
        tasks = self.plan_tasks(query)

        # If no tasks were created, query is out of scope - answer directly
        if not tasks:
            answer = self._generate_answer(query, session_outputs)
            self.logger.log_summary(answer)
            return answer

        # Main agent loop
        while any(not t.done for t in tasks):
            if step_count >= self.max_steps:
                self.logger._log("Global max steps reached — aborting to avoid runaway loop.")
                break

            task = next(t for t in tasks if not t.done)
            self.logger.log_task_start(task.description)

            per_task_steps = 0
            while per_task_steps < self.max_steps_per_task:
                if step_count >= self.max_steps:
                    self.logger._log("Global max steps reached — stopping.")
                    return

                ai_message = self.ask_for_actions(task.description, last_outputs="\n".join(session_outputs))
                
                if not ai_message.tool_calls:
                    # No tool calls means either the task is done or cannot be done with tools
                    # Always mark as done to avoid infinite loops
                    # The final answer generation will provide an appropriate response
                    task.done = True
                    self.logger.log_task_done(task.description)
                    break

                for tool_call in ai_message.tool_calls:
                    if step_count >= self.max_steps:
                        break

                    tool_name = tool_call["name"]
                    inp_args = tool_call["args"]
                    action_sig = f"{tool_name}:{inp_args}"

                    # stuck detection
                    last_actions.append(action_sig)
                    if len(last_actions) > 4:
                        last_actions = last_actions[-4:]
                    if len(set(last_actions)) == 1 and len(last_actions) == 4:
                        self.logger._log("Detected repeating action — aborting to avoid loop.")
                        return
                    
                    tool_to_run = next((t for t in TOOLS if t.name == tool_name), None)
                    if tool_to_run and self.confirm_action(tool_name, str(inp_args)):
                        try:
                            result = self._execute_tool(tool_to_run, tool_name, inp_args)
                            self.logger.log_tool_run(tool_name, f"{result}")
                            session_outputs.append(f"Output of {tool_name} with args {inp_args}: {result}")
                        except Exception as e:
                            self.logger._log(f"Tool execution failed: {e}")
                            session_outputs.append(f"Error from {tool_name} with args {inp_args}: {e}")
                    else:
                        self.logger._log(f"Invalid tool: {tool_name}")

                    step_count += 1
                    per_task_steps += 1

                # check after this batch if task seems done
                if self.ask_if_done(task.description, "\n".join(session_outputs)):
                    task.done = True
                    self.logger.log_task_done(task.description)
                    break

        # Generate answer based on all collected data
        answer = self._generate_answer(query, session_outputs)
        self.logger.log_summary(answer)
        return answer
    
    # ---------- answer generation ----------
    @show_progress("Generating answer...", "Answer ready")
    def _generate_answer(self, query: str, session_outputs: list) -> str:
        """Generate the final answer based on collected data."""
        all_results = "\n\n".join(session_outputs) if session_outputs else "No data was collected."
        answer_prompt = f"""
        Original user query: "{query}"
        
        Data and results collected from tools:
        {all_results}
        
        Based on the data above, provide a comprehensive answer to the user's query.
        Include specific numbers, calculations, and insights.
        """
        answer_obj = call_llm(answer_prompt, system_prompt=ANSWER_SYSTEM_PROMPT, output_schema=Answer)
        return answer_obj.answer
