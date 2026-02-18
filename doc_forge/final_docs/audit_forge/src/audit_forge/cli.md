Loading model... 


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b8083-ae2d3f28a
model      : qwen2.5-1.5b-instruct-q5_k_m.gguf
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read               add a text file


> <|im_start|>system
You are an Eidosian Documentation Forge agent. Your task is to generate detailed, accurate, and structurally elegant documentation for the provided source file.
Follow these requirements:
1. File Summary: High-level overview of what the file does.
2. API Documentation: List all functions, classes, and their parameters/methods.
3. Current Status: Accurate assessment of the file's current state based on its content and comments.
4. Potential Future Directions: Eidosian ideas for ... (truncated)

# Audit Forge CLI Documentation

## File Summary

The `audit_forge/cli.py` file provides a command-line interface (CLI) for managing audit data and tasks. It leverages the Eidosian Core framework to handle operations related to auditing files, marking them as reviewed, and adding todo items.

## API Documentation

### Functions
- **coverage(root: Path = DEFAULT_ROOT)**:
  - Description: Checks the audit coverage of a specified directory.
  - Parameters:
    - `root`: The root directory for audit data (default is current working directory).
  
- **mark(path: str, agent: str = "user")**:
  - Description: Marks a file as reviewed in the specified directory.
  - Parameters:
    - `path`: Path to the file to be marked as reviewed.
    - `agent`: The name of the reviewer (default is "user").

- **todo(task: str, section: str = "Immediate")**:
  - Description: Adds a todo item to the audit data.
  - Parameters:
    - `task`: The task description.
    - `section`: Section where the task should be added (default is "Immediate").

### Classes

None

## Current Status

- **Default Directory**: The CLI automatically defaults to the current working directory if no specific root path is specified. This allows for easy execution of commands without specifying a separate root path each time.
- **Eidosian Core Integration**: The CLI utilizes Eidosian Core, which provides utilities like `eidosian` and `typer`, facilitating seamless integration with other auditing tasks.
- **Command Line Interface (CLI)**: The CLI is designed to interact with the audit data directory using simple text-based commands.

## Potential Future Directions

- **Enhanced Error Handling**: Implement more robust error handling mechanisms to provide clearer feedback when commands fail or if paths are incorrect.
- **Parameter Validation**: Add validation for certain parameters, such as ensuring that `path` exists before attempting to mark it as reviewed.
- **Logging and Debugging**: Introduce logging functionality to aid in debugging issues during command execution. This could include detailed logs of all actions taken by the CLI.

---

This documentation provides a clear overview of the `audit_forge/cli.py` file, its functions and classes, current status, and potential future enhancements.

[ Prompt: 52.2 t/s | Generation: 7.1 t/s ]

Exiting...