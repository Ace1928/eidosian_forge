import sys
from tests.base import BaseTestCase
from pyasn1.compat import integer
def test_from_bytes_zero(self):
    assert 0 == integer.from_bytes('\x00', signed=False)