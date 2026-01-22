import sys
from tests.base import BaseTestCase
from pyasn1.compat import octets
def test_octs2str(self):
    assert '\x01\x02\x03' == octets.octs2str('\x01\x02\x03')