import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_encode_combines_short(self):
    self.encoder.write(_b('abc'))
    self.encoder.write(_b('def'))
    self.encoder.close()
    self.assertEqual(_b('6\r\nabcdef0\r\n'), self.output.getvalue())