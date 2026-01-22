import string
import unittest
import six
from apitools.base.py import buffered_stream
from apitools.base.py import exceptions
def testOffsetStream(self):
    bs = buffered_stream.BufferedStream(self.stream, 50, 100)
    self.assertEqual(len(self.value), len(bs))
    self.assertEqual(self.value, bs.read(len(self.value)))
    self.assertEqual(50 + len(self.value), bs.stream_end_position)