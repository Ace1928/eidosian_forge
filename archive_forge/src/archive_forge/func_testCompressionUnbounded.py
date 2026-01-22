import unittest
from apitools.base.py import compression
from apitools.base.py import gzip
import six
def testCompressionUnbounded(self):
    """Test unbounded compression.

        Test that the input stream is exhausted when length is none.
        """
    output, read, exhausted = compression.CompressStream(self.stream, None, 9)
    self.assertLess(output.length, self.length)
    self.assertEqual(read, self.length)
    self.assertTrue(exhausted)