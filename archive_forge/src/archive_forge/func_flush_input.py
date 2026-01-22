from __future__ import annotations
import asyncio
import contextvars
import os
import re
import signal
import sys
import threading
import time
from asyncio import (
from contextlib import ExitStack, contextmanager
from subprocess import Popen
from traceback import format_tb
from typing import (
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.clipboard import Clipboard, InMemoryClipboard
from prompt_toolkit.cursor_shapes import AnyCursorShapeConfig, to_cursor_shape_config
from prompt_toolkit.data_structures import Size
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.eventloop import (
from prompt_toolkit.eventloop.utils import call_soon_threadsafe
from prompt_toolkit.filters import Condition, Filter, FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.input.base import Input
from prompt_toolkit.input.typeahead import get_typeahead, store_typeahead
from prompt_toolkit.key_binding.bindings.page_navigation import (
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.key_binding.emacs_state import EmacsState
from prompt_toolkit.key_binding.key_bindings import (
from prompt_toolkit.key_binding.key_processor import KeyPressEvent, KeyProcessor
from prompt_toolkit.key_binding.vi_state import ViState
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import Container, Window
from prompt_toolkit.layout.controls import BufferControl, UIControl
from prompt_toolkit.layout.dummy import create_dummy_layout
from prompt_toolkit.layout.layout import Layout, walk
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.renderer import Renderer, print_formatted_text
from prompt_toolkit.search import SearchState
from prompt_toolkit.styles import (
from prompt_toolkit.utils import Event, in_main_thread
from .current import get_app_session, set_app
from .run_in_terminal import in_terminal, run_in_terminal
def flush_input() -> None:
    if not self.is_done:
        keys = self.input.flush_keys()
        self.key_processor.feed_multiple(keys)
        self.key_processor.process_keys()
        if self.input.closed:
            f.set_exception(EOFError)