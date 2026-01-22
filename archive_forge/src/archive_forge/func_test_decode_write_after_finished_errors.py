import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_decode_write_after_finished_errors(self):
    self.assertEqual(_b('1234'), self.decoder.write(_b('3\r\nabc0\r\n1234')))
    self.assertRaises(ValueError, self.decoder.write, _b(''))