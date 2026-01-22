import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testGetTypeByPosition(self):
    assert self.e.getTypeByPosition(0) == univ.OctetString(''), 'getTypeByPosition() fails'