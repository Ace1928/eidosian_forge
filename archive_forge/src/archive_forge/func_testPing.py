from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
def testPing(self):
    fake_hid_dev = util.FakeHidDevice(bytearray([0, 0, 0, 1]))
    t = hidtransport.UsbHidTransport(fake_hid_dev)
    reply = t.SendPing(b'1234')
    self.assertEquals(reply, b'1234')