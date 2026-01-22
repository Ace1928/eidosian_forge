import sys
from tests.base import BaseTestCase
from pyasn1.compat import octets
def test_isOctetsType(self):
    assert octets.isOctetsType('abc') == True
    assert octets.isOctetsType(123) == False
    assert octets.isOctetsType(unicode('abc')) == False