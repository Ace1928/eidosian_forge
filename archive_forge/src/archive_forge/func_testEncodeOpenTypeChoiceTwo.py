import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import char
from pyasn1.codec.ber import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testEncodeOpenTypeChoiceTwo(self):
    self.s.clear()
    self.s[0] = 2
    self.s[1] = univ.OctetString('quick brown')
    assert encoder.encode(self.s, asn1Spec=self.s) == ints2octs((48, 16, 2, 1, 2, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110))