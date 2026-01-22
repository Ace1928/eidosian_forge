import code
import os
import sys
import tempfile
import io
from typing import cast
import unittest
from contextlib import contextmanager
from functools import partial
from unittest import mock
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython.curtsiesfrontend import interpreter
from bpython.curtsiesfrontend import events as bpythonevents
from bpython.curtsiesfrontend.repl import LineType
from bpython import autocomplete
from bpython import config
from bpython import args
from bpython.test import (
from curtsies import events
from curtsies.window import CursorAwareWindow
from importlib import invalidate_caches
def test_control_events_in_large_paste(self):
    """Large paste events should ignore control characters"""
    p = events.PasteEvent()
    p.events = ['a', '<Ctrl-a>'] + ['e'] * curtsiesrepl.MAX_EVENTS_POSSIBLY_NOT_PASTE
    self.repl.process_event(p)
    self.assertEqual(self.repl.current_line, 'a' + 'e' * curtsiesrepl.MAX_EVENTS_POSSIBLY_NOT_PASTE)