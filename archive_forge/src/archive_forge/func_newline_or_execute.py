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
def newline_or_execute(event):
    """When the user presses return, insert a newline or execute the code."""
    b = event.current_buffer
    d = b.document
    if b.complete_state:
        cc = b.complete_state.current_completion
        if cc:
            b.apply_completion(cc)
        else:
            b.cancel_completion()
        return
    if d.line_count == 1:
        check_text = d.text
    else:
        check_text = d.text[:d.cursor_position]
    status, indent = shell.check_complete(check_text)
    after_cursor = d.text[d.cursor_position:]
    reformatted = False
    if not after_cursor.strip():
        reformat_text_before_cursor(b, d, shell)
        reformatted = True
    if not (d.on_last_line or d.cursor_position_row >= d.line_count - d.empty_line_count_at_the_end()):
        if shell.autoindent:
            b.insert_text('\n' + indent)
        else:
            b.insert_text('\n')
        return
    if status != 'incomplete' and b.accept_handler:
        if not reformatted:
            reformat_text_before_cursor(b, d, shell)
        b.validate_and_handle()
    elif shell.autoindent:
        b.insert_text('\n' + indent)
    else:
        b.insert_text('\n')