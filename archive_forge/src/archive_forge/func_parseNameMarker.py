from contextlib import contextmanager
from typing import Iterator, Optional, Tuple
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
def parseNameMarker(state: StateBlock, startLine: int) -> Tuple[int, str]:
    """Parse field name: `:name:`

    :returns: position after name marker, name text
    """
    start = state.bMarks[startLine] + state.tShift[startLine]
    pos = start
    maximum = state.eMarks[startLine]
    if pos + 2 >= maximum:
        return (-1, '')
    if state.src[pos] != ':':
        return (-1, '')
    name_length = 1
    found_close = False
    for ch in state.src[pos + 1:]:
        if ch == '\n':
            break
        if ch == ':':
            found_close = True
            break
        name_length += 1
    if not found_close:
        return (-1, '')
    name_text = state.src[pos + 1:pos + name_length]
    if not name_text.strip():
        return (-1, '')
    return (pos + name_length + 1, name_text)