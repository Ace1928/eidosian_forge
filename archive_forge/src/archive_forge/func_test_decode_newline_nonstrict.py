import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_decode_newline_nonstrict(self):
    """Tolerate chunk markers with no CR character."""
    self.decoder = subunit.chunked.Decoder(self.output, strict=False)
    self.assertEqual(None, self.decoder.write(_b('a\n')))
    self.assertEqual(None, self.decoder.write(_b('abcdeabcde')))
    self.assertEqual(_b(''), self.decoder.write(_b('0\n')))
    self.assertEqual(_b('abcdeabcde'), self.output.getvalue())