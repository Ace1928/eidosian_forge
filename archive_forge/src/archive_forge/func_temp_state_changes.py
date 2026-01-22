from contextlib import contextmanager
from typing import Iterator, Optional, Tuple
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
@contextmanager
def temp_state_changes(state: StateBlock, startLine: int) -> Iterator[None]:
    """Allow temporarily changing certain state attributes."""
    oldTShift = state.tShift[startLine]
    oldSCount = state.sCount[startLine]
    oldBlkIndent = state.blkIndent
    oldSrc = state.src
    yield
    state.blkIndent = oldBlkIndent
    state.tShift[startLine] = oldTShift
    state.sCount[startLine] = oldSCount
    state.src = oldSrc