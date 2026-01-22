from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def test_closed_shouldnt_raise_on_detached_stream(self):
    stream = TextIOWrapper(StringIO())
    stream.detach()
    wrapper = StreamWrapper(stream, None)
    self.assertEqual(wrapper.closed, True)