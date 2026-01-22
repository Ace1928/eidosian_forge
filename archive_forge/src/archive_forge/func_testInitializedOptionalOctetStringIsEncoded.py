import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.der import encoder
from pyasn1.compat.octets import ints2octs
def testInitializedOptionalOctetStringIsEncoded(self):
    self.s = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.OptionalNamedType('str', univ.OctetString())))
    self.s.clear()
    self.s[0] = ''
    assert encoder.encode(self.s) == ints2octs((48, 2, 4, 0))