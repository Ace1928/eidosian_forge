from __future__ import unicode_literals
from .auto_suggest import AutoSuggest
from .clipboard import ClipboardData
from .completion import Completer, Completion, CompleteEvent
from .document import Document
from .enums import IncrementalSearchDirection
from .filters import to_simple_filter
from .history import History, InMemoryHistory
from .search_state import SearchState
from .selection import SelectionType, SelectionState, PasteMode
from .utils import Event
from .cache import FastDictCache
from .validation import ValidationError
from six.moves import range
import os
import re
import six
import subprocess
import tempfile
def document_for_search(self, search_state):
    """
        Return a :class:`~prompt_toolkit.document.Document` instance that has
        the text/cursor position for this search, if we would apply it. This
        will be used in the
        :class:`~prompt_toolkit.layout.controls.BufferControl` to display
        feedback while searching.
        """
    search_result = self._search(search_state, include_current_position=True)
    if search_result is None:
        return self.document
    else:
        working_index, cursor_position = search_result
        if working_index == self.working_index:
            selection = self.selection_state
        else:
            selection = None
        return Document(self._working_lines[working_index], cursor_position, selection=selection)