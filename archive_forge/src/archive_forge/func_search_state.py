from __future__ import annotations
import time
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, Hashable, Iterable, NamedTuple
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import (
from prompt_toolkit.lexers import Lexer, SimpleLexer
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.search import SearchState
from prompt_toolkit.selection import SelectionType
from prompt_toolkit.utils import get_cwidth
from .processors import (
@property
def search_state(self) -> SearchState:
    """
        Return the `SearchState` for searching this `BufferControl`. This is
        always associated with the search control. If one search bar is used
        for searching multiple `BufferControls`, then they share the same
        `SearchState`.
        """
    search_buffer_control = self.search_buffer_control
    if search_buffer_control:
        return search_buffer_control.searcher_search_state
    else:
        return SearchState()