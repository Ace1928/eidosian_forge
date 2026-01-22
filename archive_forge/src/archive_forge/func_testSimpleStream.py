import unittest
from apitools.base.py import compression
from apitools.base.py import gzip
import six
def testSimpleStream(self):
    """Test simple stream operations.

        Test that the stream can be written to and read from. Also test that
        reading from the stream consumes the bytes.
        """
    self.assertEqual(self.stream.length, 0)
    self.stream.write(b'Sample data')
    self.assertEqual(self.stream.length, 11)
    data = self.stream.read(11)
    self.assertEqual(data, b'Sample data')
    self.assertEqual(self.stream.length, 0)