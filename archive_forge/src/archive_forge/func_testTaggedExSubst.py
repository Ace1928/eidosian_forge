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
def testTaggedExSubst(self):
    assert decoder.decode(ints2octs((164, 5, 4, 3, 102, 111, 120)), asn1Spec=self.s, substrateFun=lambda a, b, c: (b, b[c:])) == (ints2octs((164, 5, 4, 3, 102, 111, 120)), str2octs(''))