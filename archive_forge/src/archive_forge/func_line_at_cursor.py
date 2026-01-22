from collections import namedtuple
from io import StringIO
from keyword import iskeyword
import tokenize
from tokenize import TokenInfo
from typing import List, Optional
def line_at_cursor(cell, cursor_pos=0):
    """Return the line in a cell at a given cursor position

    Used for calling line-based APIs that don't support multi-line input, yet.

    Parameters
    ----------
    cell : str
        multiline block of text
    cursor_pos : integer
        the cursor position

    Returns
    -------
    (line, offset): (string, integer)
        The line with the current cursor, and the character offset of the start of the line.
    """
    offset = 0
    lines = cell.splitlines(True)
    for line in lines:
        next_offset = offset + len(line)
        if not line.endswith('\n'):
            next_offset += 1
        if next_offset > cursor_pos:
            break
        offset = next_offset
    else:
        line = ''
    return (line, offset)