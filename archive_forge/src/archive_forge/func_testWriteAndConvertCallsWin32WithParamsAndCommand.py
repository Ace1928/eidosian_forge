from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def testWriteAndConvertCallsWin32WithParamsAndCommand(self):
    stream = AnsiToWin32(Mock())
    stream.convert = True
    stream.call_win32 = Mock()
    stream.extract_params = Mock(return_value='params')
    data = {'abc\x1b[adef': ('a', 'params'), 'abc\x1b[;;bdef': ('b', 'params'), 'abc\x1b[0cdef': ('c', 'params'), 'abc\x1b[;;0;;Gdef': ('G', 'params'), 'abc\x1b[1;20;128Hdef': ('H', 'params')}
    for datum, expected in data.items():
        stream.call_win32.reset_mock()
        stream.write_and_convert(datum)
        self.assertEqual(stream.call_win32.call_args[0], expected)