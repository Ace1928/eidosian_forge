from contextlib import contextmanager
from typing import Iterator, Optional, Tuple
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from mdit_py_plugins.utils import is_code_block
@contextmanager
def set_parent_type(state: StateBlock, name: str) -> Iterator[None]:
    """Temporarily set parent type to `name`"""
    oldParentType = state.parentType
    state.parentType = name
    yield
    state.parentType = oldParentType