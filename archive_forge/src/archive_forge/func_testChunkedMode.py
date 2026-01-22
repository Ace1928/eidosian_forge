import sys
from tests.base import BaseTestCase
from pyasn1.codec.der import decoder
from pyasn1.compat.octets import ints2octs, null
from pyasn1.error import PyAsn1Error
def testChunkedMode(self):
    try:
        decoder.decode(ints2octs((36, 23, 4, 2, 81, 117, 4, 2, 105, 99, 4, 2, 107, 32, 4, 2, 98, 114, 4, 2, 111, 119, 4, 1, 110)))
    except PyAsn1Error:
        pass
    else:
        assert 0, 'chunked encoding tolerated'