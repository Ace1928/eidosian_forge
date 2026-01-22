import sys
from tests.base import BaseTestCase
from pyasn1.compat import octets
def test_isStringType(self):
    assert octets.isStringType('abc') == True
    assert octets.isStringType(123) == False
    assert octets.isStringType(unicode('abc')) == True