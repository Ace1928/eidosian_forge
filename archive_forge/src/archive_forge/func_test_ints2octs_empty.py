import sys
from tests.base import BaseTestCase
from pyasn1.compat import octets
def test_ints2octs_empty(self):
    assert not octets.ints2octs([])