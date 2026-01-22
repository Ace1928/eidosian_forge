import ctypes
import sys
import mock
from pyu2f import errors
from pyu2f.hid import macos
@mock.patch.object(macos.threading, 'Thread')
@mock.patch.multiple(macos, iokit=mock.DEFAULT, cf=mock.DEFAULT, GetDeviceIntProperty=mock.DEFAULT)
def testCallReadSuccess(self, thread, iokit, cf, GetDeviceIntProperty):
    init_mock_iokit(iokit)
    init_mock_cf(cf)
    init_mock_get_int_property(GetDeviceIntProperty)
    device = macos.MacOsHidDevice('fakepath')
    report = (ctypes.c_uint8 * 64)()
    report[:] = range(64)[:]
    q = device.read_queue
    macos.HidReadCallback(q, None, None, None, 0, report, 64)
    read_result = device.Read()
    self.assertEquals(read_result, list(range(64)), 'Read data should match data passed into the callback')