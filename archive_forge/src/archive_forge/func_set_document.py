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
def set_document(self, value, bypass_readonly=False):
    """
        Set :class:`~prompt_toolkit.document.Document` instance. Like the
        ``document`` property, but accept an ``bypass_readonly`` argument.

        :param bypass_readonly: When True, don't raise an
                                :class:`.EditReadOnlyBuffer` exception, even
                                when the buffer is read-only.
        """
    assert isinstance(value, Document)
    if not bypass_readonly and self.read_only():
        raise EditReadOnlyBuffer()
    text_changed = self._set_text(value.text)
    cursor_position_changed = self._set_cursor_position(value.cursor_position)
    if text_changed:
        self._text_changed()
    if cursor_position_changed:
        self._cursor_position_changed()