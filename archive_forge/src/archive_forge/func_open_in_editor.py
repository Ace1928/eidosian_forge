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
def open_in_editor(self, cli):
    """
        Open code in editor.

        :param cli: :class:`~prompt_toolkit.interface.CommandLineInterface`
            instance.
        """
    if self.read_only():
        raise EditReadOnlyBuffer()
    descriptor, filename = tempfile.mkstemp(self.tempfile_suffix)
    os.write(descriptor, self.text.encode('utf-8'))
    os.close(descriptor)
    succes = cli.run_in_terminal(lambda: self._open_file_in_editor(filename))
    if succes:
        with open(filename, 'rb') as f:
            text = f.read().decode('utf-8')
            if text.endswith('\n'):
                text = text[:-1]
            self.document = Document(text=text, cursor_position=len(text))
    os.remove(filename)