import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_decode_short_header(self):
    self.assertRaises(ValueError, self.decoder.write, _b('\n'))