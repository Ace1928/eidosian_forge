import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def code_to_line(code: str, cursor_pos: int) -> Tuple[str, int]:
    """Turn a multiline code block and cursor position into a single line
    and new cursor position.

    For adapting ``complete_`` and ``object_info_request``.
    """
    if not code:
        return ('', 0)
    for line in code.splitlines(True):
        n = len(line)
        if cursor_pos > n:
            cursor_pos -= n
        else:
            break
    return (line, cursor_pos)