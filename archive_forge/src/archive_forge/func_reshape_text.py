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
def reshape_text(buffer, from_row, to_row):
    """
    Reformat text, taking the width into account.
    `to_row` is included.
    (Vi 'gq' operator.)
    """
    lines = buffer.text.splitlines(True)
    lines_before = lines[:from_row]
    lines_after = lines[to_row + 1:]
    lines_to_reformat = lines[from_row:to_row + 1]
    if lines_to_reformat:
        length = re.search('^\\s*', lines_to_reformat[0]).end()
        indent = lines_to_reformat[0][:length].replace('\n', '')
        words = ''.join(lines_to_reformat).split()
        width = (buffer.text_width or 80) - len(indent)
        reshaped_text = [indent]
        current_width = 0
        for w in words:
            if current_width:
                if len(w) + current_width + 1 > width:
                    reshaped_text.append('\n')
                    reshaped_text.append(indent)
                    current_width = 0
                else:
                    reshaped_text.append(' ')
                    current_width += 1
            reshaped_text.append(w)
            current_width += len(w)
        if reshaped_text[-1] != '\n':
            reshaped_text.append('\n')
        buffer.document = Document(text=''.join(lines_before + reshaped_text + lines_after), cursor_position=len(''.join(lines_before + reshaped_text)))