import ctypes
import sys
import mock
from pyu2f import errors
from pyu2f.hid import macos
def init_mock_cf(mock_cf):
    mock_cf.CFGetTypeID = mock.Mock(return_value=123)
    mock_cf.CFNumberGetTypeID = mock.Mock(return_value=123)
    mock_cf.CFStringGetTypeID = mock.Mock(return_value=123)