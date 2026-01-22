from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def test_osc_codes(self):
    mockStdout = Mock()
    stream = AnsiToWin32(mockStdout, convert=True)
    with patch('colorama.ansitowin32.winterm') as winterm:
        data = ['\x1b]0\x07', '\x1b]0;foo\x08', '\x1b]0;colorama_test_title\x07', '\x1b]1;colorama_test_title\x07', '\x1b]2;colorama_test_title\x07', '\x1b]' + ';' * 64 + '\x08']
        for code in data:
            stream.write(code)
        self.assertEqual(winterm.set_title.call_count, 2)