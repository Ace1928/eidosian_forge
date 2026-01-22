import unittest
from apitools.base.py import compression
from apitools.base.py import gzip
import six
def testCompressionPartial(self):
    """Test partial compression.

        Test that the length parameter works correctly. The amount of data
        that's compressed can be greater than or equal to the requested length.
        """
    output_length = 40
    output, _, exhausted = compression.CompressStream(self.stream, output_length, 9)
    self.assertLessEqual(output_length, output.length)
    self.assertFalse(exhausted)