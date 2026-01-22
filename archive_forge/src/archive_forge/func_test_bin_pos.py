import sys
from tests.base import BaseTestCase
from pyasn1.compat import binary
def test_bin_pos(self):
    assert '0b1000000010000000100000001' == binary.bin(16843009)