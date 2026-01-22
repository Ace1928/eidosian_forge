from six.moves import range
import sys
import mock
from pyu2f import errors
from pyu2f import hidtransport
from pyu2f.tests.lib import util
def testInitPacketShape(self):
    packet = hidtransport.UsbHidTransport.InitPacket(64, bytearray(b'\x00\x00\x00\x01'), 131, 2, bytearray(b'\x01\x02'))
    self.assertEquals(packet.ToWireFormat(), RPad([0, 0, 0, 1, 131, 0, 2, 1, 2], 64))
    copy = hidtransport.UsbHidTransport.InitPacket.FromWireFormat(64, packet.ToWireFormat())
    self.assertEquals(copy.cid, bytearray(b'\x00\x00\x00\x01'))
    self.assertEquals(copy.cmd, 131)
    self.assertEquals(copy.size, 2)
    self.assertEquals(copy.payload, bytearray(b'\x01\x02'))