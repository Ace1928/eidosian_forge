import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testGetPositionNearType(self):
    assert self.e.getPositionNearType(univ.OctetString.tagSet, 0) == 0
    assert self.e.getPositionNearType(univ.Integer.tagSet, 1) == 1
    assert self.e.getPositionNearType(univ.OctetString.tagSet, 2) == 2