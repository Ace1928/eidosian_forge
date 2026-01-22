import unittest
from apitools.base.py import compression
from apitools.base.py import gzip
import six
def testCompressionExhausted(self):
    """Test full compression.

        Test that highly compressible data is actually compressed in entirety.
        """
    output, read, exhausted = compression.CompressStream(self.stream, self.length, 9)
    self.assertLess(output.length, self.length)
    self.assertEqual(read, self.length)
    self.assertTrue(exhausted)