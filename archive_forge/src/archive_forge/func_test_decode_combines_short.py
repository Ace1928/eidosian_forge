import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_decode_combines_short(self):
    self.assertEqual(_b(''), self.decoder.write(_b('6\r\nabcdef0\r\n')))
    self.assertEqual(_b('abcdef'), self.output.getvalue())