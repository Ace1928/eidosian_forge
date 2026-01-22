import sys
from tests.base import BaseTestCase
from pyasn1.codec.cer import decoder
from pyasn1.compat.octets import ints2octs, str2octs, null
from pyasn1.error import PyAsn1Error
def testOverflow(self):
    try:
        decoder.decode(ints2octs((1, 2, 0, 0)))
    except PyAsn1Error:
        pass