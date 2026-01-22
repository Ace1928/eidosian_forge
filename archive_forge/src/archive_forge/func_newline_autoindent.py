import os
import signal
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Any, Optional, List
from prompt_toolkit.application.current import get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands as nc
from prompt_toolkit.key_binding.bindings.completion import (
from prompt_toolkit.key_binding.vi_state import InputMode, ViState
from prompt_toolkit.filters import Condition
from IPython.core.getipython import get_ipython
from IPython.terminal.shortcuts import auto_match as match
from IPython.terminal.shortcuts import auto_suggest
from IPython.terminal.shortcuts.filters import filter_from_string
from IPython.utils.decorators import undoc
from prompt_toolkit.enums import DEFAULT_BUFFER
def newline_autoindent(event):
    """Insert a newline after the cursor indented appropriately.

    Fancier version of former ``newline_with_copy_margin`` which should
    compute the correct indentation of the inserted line. That is to say, indent
    by 4 extra space after a function definition, class definition, context
    manager... And dedent by 4 space after ``pass``, ``return``, ``raise ...``.
    """
    shell = get_ipython()
    inputsplitter = shell.input_transformer_manager
    b = event.current_buffer
    d = b.document
    if b.complete_state:
        b.cancel_completion()
    text = d.text[:d.cursor_position] + '\n'
    _, indent = inputsplitter.check_complete(text)
    b.insert_text('\n' + ' ' * (indent or 0), move_cursor=False)