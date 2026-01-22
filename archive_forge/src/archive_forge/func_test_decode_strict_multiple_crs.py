import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_decode_strict_multiple_crs(self):
    self.assertRaises(ValueError, self.decoder.write, _b('a\r\r\n'))