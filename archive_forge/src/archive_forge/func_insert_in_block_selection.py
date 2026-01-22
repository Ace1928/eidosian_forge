from __future__ import annotations
import codecs
import string
from enum import Enum
from itertools import accumulate
from typing import Callable, Iterable, Tuple, TypeVar
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer, indent, reshape_text, unindent
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.document import Document
from prompt_toolkit.filters import (
from prompt_toolkit.filters.app import (
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.key_binding.digraphs import DIGRAPHS
from prompt_toolkit.key_binding.key_processor import KeyPress, KeyPressEvent
from prompt_toolkit.key_binding.vi_state import CharacterFind, InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.search import SearchDirection
from prompt_toolkit.selection import PasteMode, SelectionState, SelectionType
from ..key_bindings import ConditionalKeyBindings, KeyBindings, KeyBindingsBase
from .named_commands import get_by_name
@handle('I', filter=in_block_selection & ~is_read_only)
def insert_in_block_selection(event: E, after: bool=False) -> None:
    """
        Insert in block selection mode.
        """
    buff = event.current_buffer
    positions = []
    if after:

        def get_pos(from_to: tuple[int, int]) -> int:
            return from_to[1]
    else:

        def get_pos(from_to: tuple[int, int]) -> int:
            return from_to[0]
    for i, from_to in enumerate(buff.document.selection_ranges()):
        positions.append(get_pos(from_to))
        if i == 0:
            buff.cursor_position = get_pos(from_to)
    buff.multiple_cursor_positions = positions
    event.app.vi_state.input_mode = InputMode.INSERT_MULTIPLE
    buff.exit_selection()