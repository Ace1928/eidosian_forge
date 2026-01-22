import unittest
from apitools.base.py import compression
from apitools.base.py import gzip
import six
def testPartialReads(self):
    """Test partial stream reads.

        Test that the stream can be read in chunks while perserving the
        consumption mechanics.
        """
    self.stream.write(b'Sample data')
    data = self.stream.read(6)
    self.assertEqual(data, b'Sample')
    self.assertEqual(self.stream.length, 5)
    data = self.stream.read(5)
    self.assertEqual(data, b' data')
    self.assertEqual(self.stream.length, 0)