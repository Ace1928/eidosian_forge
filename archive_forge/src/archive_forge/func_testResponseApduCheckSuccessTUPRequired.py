from six.moves import range
import sys
from pyu2f import apdu
from pyu2f import errors
def testResponseApduCheckSuccessTUPRequired(self):
    resp = apdu.ResponseApdu(bytearray([105, 133]))
    self.assertRaises(errors.TUPRequiredError, resp.CheckSuccessOrRaise)