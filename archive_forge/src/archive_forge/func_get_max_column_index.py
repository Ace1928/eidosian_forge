from __future__ import annotations
from asyncio import FIRST_COMPLETED, Future, ensure_future, sleep, wait
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.data_structures import Point, Size
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, to_formatted_text
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Char, Screen, WritePosition
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.styles import (
def get_max_column_index(row: dict[int, Char]) -> int:
    """
        Return max used column index, ignoring whitespace (without style) at
        the end of the line. This is important for people that copy/paste
        terminal output.

        There are two reasons we are sometimes seeing whitespace at the end:
        - `BufferControl` adds a trailing space to each line, because it's a
          possible cursor position, so that the line wrapping won't change if
          the cursor position moves around.
        - The `Window` adds a style class to the current line for highlighting
          (cursor-line).
        """
    numbers = (index for index, cell in row.items() if cell.char != ' ' or style_string_has_style[cell.style])
    return max(numbers, default=0)