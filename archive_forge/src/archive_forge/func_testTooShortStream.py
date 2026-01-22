import string
import unittest
import six
from apitools.base.py import exceptions
from apitools.base.py import stream_slice
def testTooShortStream(self):
    ss = stream_slice.StreamSlice(self.stream, 1000)
    self.assertEqual(self.value, ss.read())
    self.assertEqual('', ss.read(0))
    with self.assertRaises(exceptions.StreamExhausted) as e:
        ss.read()
    with self.assertRaises(exceptions.StreamExhausted) as e:
        ss.read(10)
    self.assertIn('exhausted after %d' % len(self.value), str(e.exception))