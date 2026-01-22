from six.moves import range
import sys
from pyu2f import apdu
from pyu2f import errors
def testResponseApduCheckSuccessOtherError(self):
    resp = apdu.ResponseApdu(bytearray([250, 5]))
    self.assertRaises(errors.ApduError, resp.CheckSuccessOrRaise)