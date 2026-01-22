from typing import Dict, List
from nbformat import NotebookNode
@classmethod
def error_from_timeout_and_cell(cls, msg: str, timeout: int, cell: NotebookNode) -> 'CellTimeoutError':
    """Create an error from a timeout on a cell."""
    if cell and cell.source:
        src_by_lines = cell.source.strip().split('\n')
        src = cell.source if len(src_by_lines) < 11 else f'{src_by_lines[:5]}\n...\n{src_by_lines[-5:]}'
    else:
        src = 'Cell contents not found.'
    return cls(timeout_err_msg.format(timeout=timeout, msg=msg, cell_contents=src))