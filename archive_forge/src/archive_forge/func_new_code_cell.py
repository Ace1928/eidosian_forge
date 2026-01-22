from __future__ import annotations
from nbformat._struct import Struct
def new_code_cell(input=None, prompt_number=None, outputs=None, language='python', collapsed=False):
    """Create a new code cell with input and output"""
    cell = NotebookNode()
    cell.cell_type = 'code'
    if language is not None:
        cell.language = str(language)
    if input is not None:
        cell.input = str(input)
    if prompt_number is not None:
        cell.prompt_number = int(prompt_number)
    if outputs is None:
        cell.outputs = []
    else:
        cell.outputs = outputs
    if collapsed is not None:
        cell.collapsed = bool(collapsed)
    return cell