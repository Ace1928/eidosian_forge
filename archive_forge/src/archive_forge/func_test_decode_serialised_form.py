import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_decode_serialised_form(self):
    self.assertEqual(None, self.decoder.write(_b('F\r\n')))
    self.assertEqual(None, self.decoder.write(_b('serialised\n')))
    self.assertEqual(_b(''), self.decoder.write(_b('form0\r\n')))