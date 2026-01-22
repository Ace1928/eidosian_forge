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
def testExplicitTag(self):
    s = self.s.subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 4))
    s.setComponentByPosition(0, univ.Null(null))
    assert decoder.decode(ints2octs((164, 2, 5, 0)), asn1Spec=s) == (s, null)