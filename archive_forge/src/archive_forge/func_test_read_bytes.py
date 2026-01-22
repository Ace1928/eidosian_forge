import io
import time
import unittest
from fastimport import (
from :2
def test_read_bytes(self):
    s = io.BytesIO(b'foo\nbar\nbaz\n')
    p = parser.LineBasedParser(s)
    self.assertEqual(b'fo', p.read_bytes(2))
    self.assertEqual(b'o\nb', p.read_bytes(3))
    self.assertEqual(b'ar', p.next_line())
    p.push_line(b'bar')
    self.assertEqual(b'baz', p.read_bytes(3))
    self.assertRaises(errors.MissingBytes, p.read_bytes, 10)