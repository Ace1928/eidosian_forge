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
def complete_previous(self, count=1, disable_wrap_around=False):
    """
        Browse to the previous completions.
        (Does nothing if there are no completion.)
        """
    if self.complete_state:
        if self.complete_state.complete_index == 0:
            index = None
            if disable_wrap_around:
                return
        elif self.complete_state.complete_index is None:
            index = len(self.complete_state.current_completions) - 1
        else:
            index = max(0, self.complete_state.complete_index - count)
        self.go_to_completion(index)