import sys
from tests.base import BaseTestCase
from pyasn1.compat import octets
def test_ints2octs(self):
    assert '\x01\x02\x03' == octets.ints2octs([1, 2, 3])