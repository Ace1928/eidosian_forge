import ctypes
import sys
import mock
from pyu2f import errors
from pyu2f.hid import macos
@mock.patch.object(macos.threading, 'Thread')
@mock.patch.multiple(macos, iokit=mock.DEFAULT, cf=mock.DEFAULT, GetDeviceIntProperty=mock.DEFAULT)
def testCallWriteFailure(self, thread, iokit, cf, GetDeviceIntProperty):
    init_mock_iokit(iokit)
    init_mock_cf(cf)
    init_mock_get_int_property(GetDeviceIntProperty)
    iokit.IOHIDDeviceSetReport.return_value = -1
    device = macos.MacOsHidDevice('fakepath')
    data = bytearray(range(64))
    with self.assertRaises(errors.OsHidError):
        device.Write(data)