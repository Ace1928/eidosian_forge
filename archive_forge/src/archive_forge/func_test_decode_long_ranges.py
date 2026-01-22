import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_decode_long_ranges(self):
    self.assertEqual(None, self.decoder.write(_b('10000\r\n')))
    self.assertEqual(None, self.decoder.write(_b('1' * 65536)))
    self.assertEqual(None, self.decoder.write(_b('10000\r\n')))
    self.assertEqual(None, self.decoder.write(_b('2' * 65536)))
    self.assertEqual(_b(''), self.decoder.write(_b('0\r\n')))
    self.assertEqual(_b('1' * 65536 + '2' * 65536), self.output.getvalue())