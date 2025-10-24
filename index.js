#!/usr/bin/env node

const { spawn } = require('child_process');

const child = spawn('stockanalysis', process.argv.slice(2), {
    stdio: 'inherit',
    shell: true
});

child.on('close', (code) => {
    process.exit(code);
});
