from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
@patch('colorama.ansitowin32.winterm', None)
@patch('colorama.ansitowin32.winapi_test', lambda *_: True)
def testStripIsTrueOnWindows(self):
    with osname('nt'):
        mockStdout = Mock()
        stream = AnsiToWin32(mockStdout)
        self.assertTrue(stream.strip)