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
def testDontDecodeOpenTypesChoiceOne(self):
    s, r = decoder.decode(ints2octs((48, 6, 2, 1, 1, 2, 1, 12)), asn1Spec=self.s)
    assert not r
    assert s[0] == 1
    assert s[1] == ints2octs((2, 1, 12))