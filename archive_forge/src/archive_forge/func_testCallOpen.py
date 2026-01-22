import base64
import os
import sys
import mock
from pyu2f.hid import linux
def testCallOpen(self):
    AddDevice(self.fs, 'hidraw1', 'Yubico U2F', 4176, 1031, YUBICO_RD)
    fake_open = fake_filesystem.FakeFileOpen(self.fs)
    with mock.patch.object(py_builtins, 'open', fake_open):
        fake_dev_os = FakeDeviceOsModule()
        with mock.patch.object(linux, 'os', fake_dev_os):
            dev = linux.LinuxHidDevice('/dev/hidraw1')
            self.assertEquals(dev.GetInReportDataLength(), 64)
            self.assertEquals(dev.GetOutReportDataLength(), 64)
            dev.Write(list(range(0, 64)))
            self.assertEquals(list(fake_dev_os.data_written), [0] + list(range(0, 64)))
            fake_dev_os.data_to_return = b'x' * 64
            self.assertEquals(dev.Read(), [120] * 64)