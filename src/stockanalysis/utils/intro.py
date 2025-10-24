def print_intro():
    """Display the welcome screen with ASCII art."""
    # ANSI color codes
    LIGHT_BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Clear screen effect with some spacing
    print("\n" * 2)
    
    # Welcome box with light blue border
    box_width = 50
    welcome_text = "Welcome to StockAnalysis"
    padding = (box_width - len(welcome_text) - 2) // 2
    
    print(f"{LIGHT_BLUE}{'═' * box_width}{RESET}")
    print(f"{LIGHT_BLUE}║{' ' * padding}{BOLD}{welcome_text}{RESET}{LIGHT_BLUE}{' ' * (box_width - len(welcome_text) - padding - 2)}║{RESET}")
    print(f"{LIGHT_BLUE}{'═' * box_width}{RESET}")
    print()
    
    # ASCII art for StockAnalysis in block letters (financial terminal style)
    stockanalysis_art = f"""{BOLD}{LIGHT_BLUE}
  _____ _             _             _             _     _           
 / ____| |           | |           | |           | |   (_)          
| (___ | |_ ___   ___| | ___   ___ | | __ _ _ __ | |__  _  ___ ___ 
 \\___ \\| __/ _ \\ / __| |/ _ \\ / _ \\| |/ _` | '_ \\| '_ \\| |/ __/ __|
 ____) | || (_) | (__| | (_) | (_) | | (_| | | | | | | | | (__\\__ \\
|_____/ \\__\\___/ \\___|_|\\___/ \\___/|_|\\__,_|_| |_|_| |_|_|\\___|___/
{RESET}"""
    
    print(stockanalysis_art)
    print()
    print("Your AI assistant for financial analysis.")
    print("Ask me any questions. Type 'exit' or 'quit' to end.")
    print()
