import ctypes
import sys
import mock
from pyu2f import errors
from pyu2f.hid import macos
def init_mock_iokit(mock_iokit):
    mock_iokit.IOHIDDeviceOpen = mock.Mock(return_value=0)
    mock_iokit.IOHIDDeviceSetReport = mock.Mock(return_value=0)
    mock_iokit.IOHIDDeviceCreate = mock.Mock(return_value='handle')