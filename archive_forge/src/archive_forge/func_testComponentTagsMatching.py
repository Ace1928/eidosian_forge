import math
import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import univ
from pyasn1.type import tag
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import error
from pyasn1.compat.octets import str2octs, ints2octs, octs2ints
from pyasn1.error import PyAsn1Error
def testComponentTagsMatching(self):
    s = self.s1.clone()
    s.strictConstraints = True
    o = univ.OctetString('abc').subtype(explicitTag=tag.Tag(tag.tagClassPrivate, tag.tagFormatSimple, 12))
    try:
        s.setComponentByName('name', o)
    except PyAsn1Error:
        pass
    else:
        assert 0, 'inner supertype tag allowed'