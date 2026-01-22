import ast  # Importing the abstract syntax tree module to manipulate and analyze Python abstract syntax grammatically. Documentation: https://docs.python.org/3/library/ast.html
import json  # Importing the JSON module to encode and decode JSON data. Documentation: https://docs.python.org/3/library/json.html
import tkinter as tk  # Importing the tkinter module for creating graphical user interfaces. Documentation: https://docs.python.org/3/library/tkinter.html
from tkinter import (
import os  # Importing the os module to interact with the operating system. Documentation: https://docs.python.org/3/library/os.html
import logging  # Importing the logging module to enable logging capabilities. Documentation: https://docs.python.org/3/library/logging.html
from typing import (
import docstring_parser  # Importing the docstring_parser module to parse Python docstrings. Documentation: https://pypi.org/project/docstring-parser/
from concurrent.futures import (
def generate_documentation() -> None:
    """
        Orchestrates the documentation generation process by interacting with the user to select source files and a directory,
        and then processing those files to generate documentation.

        This function performs several key operations:
        1. Invokes the `browse_files` function to allow the user to select Python source files.
        2. Processes the selected files using `robust_parse_files` to extract necessary documentation data.
        3. Asks the user to select a directory for saving the generated documentation through `save_directory`.
        4. Saves the processed data into a JSON format in the chosen directory using `save_json`.
        5. Handles any user cancellations during file or directory selection with appropriate error messages.

        Raises:
            - If no files are selected, it raises a user alert and logs a warning.
            - If no directory is selected after files have been processed, it raises a user alert and logs a warning.

        Returns:
            None: This function does not return any value but triggers file I/O operations.

        Examples:
            - If the user selects Python files and chooses a directory, the documentation is generated and saved.
            - If the user cancels the file selection, an error message pops up and a warning is logged.

        Note:
            This function is dependent on the successful execution of `browse_files`, `robust_parse_files`, `save_directory`, and `save_json`.
            Any exceptions raised by these functions must be handled where they occur.
        """
    src_files: List[str] = browse_files()
    if src_files:
        data: List[Dict[str, Any]] = robust_parse_files(src_files)
        if data:
            output_dir: str = save_directory()
            if output_dir:
                save_json(data, output_dir)
            else:
                messagebox.showerror('Error', 'No save directory chosen.')
                logging.warning('No save directory chosen.')
    else:
        messagebox.showerror('Error', 'No files chosen.')
        logging.warning('No files chosen.')