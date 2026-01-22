from six.moves import range
import sys
from pyu2f import apdu
from pyu2f import errors
def testSerializeCommandApdu(self):
    cmd = apdu.CommandApdu(0, 1, 3, 4, bytearray([16, 32, 48]))
    self.assertEqual(cmd.ToByteArray(), bytearray([0, 1, 3, 4, 0, 0, 3, 16, 32, 48, 0, 0]))
    self.assertEqual(cmd.ToLegacyU2FByteArray(), bytearray([0, 1, 3, 4, 0, 0, 3, 16, 32, 48, 0, 0]))