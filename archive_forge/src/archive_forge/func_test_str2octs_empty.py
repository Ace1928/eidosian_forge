import sys
from tests.base import BaseTestCase
from pyasn1.compat import octets
def test_str2octs_empty(self):
    assert not octets.str2octs('')