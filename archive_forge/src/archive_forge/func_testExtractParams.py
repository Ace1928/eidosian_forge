from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def testExtractParams(self):
    stream = AnsiToWin32(Mock())
    data = {'': (0,), ';;': (0,), '2': (2,), ';;002;;': (2,), '0;1': (0, 1), ';;003;;456;;': (3, 456), '11;22;33;44;55': (11, 22, 33, 44, 55)}
    for datum, expected in data.items():
        self.assertEqual(stream.extract_params('m', datum), expected)