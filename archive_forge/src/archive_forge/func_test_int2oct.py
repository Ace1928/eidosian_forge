import sys
from tests.base import BaseTestCase
from pyasn1.compat import octets
def test_int2oct(self):
    assert '\x0c' == octets.int2oct(12)