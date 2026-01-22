from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def assert_autoresets(self, convert, autoreset=True):
    stream = AnsiToWin32(Mock())
    stream.convert = convert
    stream.reset_all = Mock()
    stream.autoreset = autoreset
    stream.winterm = Mock()
    stream.write('abc')
    self.assertEqual(stream.reset_all.called, autoreset)