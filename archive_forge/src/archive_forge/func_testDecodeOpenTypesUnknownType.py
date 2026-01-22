import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import char
from pyasn1.codec.ber import decoder
from pyasn1.codec.ber import eoo
from pyasn1.compat.octets import ints2octs, str2octs, null
from pyasn1.error import PyAsn1Error
def testDecodeOpenTypesUnknownType(self):
    try:
        s, r = decoder.decode(ints2octs((48, 6, 2, 1, 2, 6, 1, 39)), asn1Spec=self.s, decodeOpenTypes=True)
    except PyAsn1Error:
        pass
    else:
        assert False, 'unknown open type tolerated'