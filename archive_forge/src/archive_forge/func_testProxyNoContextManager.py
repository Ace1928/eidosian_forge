from io import StringIO, TextIOWrapper
from unittest import TestCase, main
from ..ansitowin32 import AnsiToWin32, StreamWrapper
from ..win32 import ENABLE_VIRTUAL_TERMINAL_PROCESSING
from .utils import osname
def testProxyNoContextManager(self):
    mockStream = MagicMock()
    mockStream.__enter__.side_effect = AttributeError()
    mockConverter = Mock()
    with self.assertRaises(AttributeError) as excinfo:
        with StreamWrapper(mockStream, mockConverter) as wrapper:
            wrapper.write('hello')