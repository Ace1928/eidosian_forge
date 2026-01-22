from __future__ import annotations
import contextvars
import datetime
import functools
import os
import signal
import threading
import traceback
from typing import (
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_session
from prompt_toolkit.filters import Condition, is_done, renderer_height_is_known
from prompt_toolkit.formatted_text import (
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import (
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.dimension import AnyDimension, D
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.styles import BaseStyle
from prompt_toolkit.utils import in_main_thread
from .formatters import Formatter, create_default_formatters
def create_key_bindings(cancel_callback: Callable[[], None] | None) -> KeyBindings:
    """
    Key bindings handled by the progress bar.
    (The main thread is not supposed to handle any key bindings.)
    """
    kb = KeyBindings()

    @kb.add('c-l')
    def _clear(event: E) -> None:
        event.app.renderer.clear()
    if cancel_callback is not None:

        @kb.add('c-c')
        def _interrupt(event: E) -> None:
            """Kill the 'body' of the progress bar, but only if we run from the main thread."""
            assert cancel_callback is not None
            cancel_callback()
    return kb