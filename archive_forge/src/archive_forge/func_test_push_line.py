import io
import time
import unittest
from fastimport import (
from :2
def test_push_line(self):
    s = io.BytesIO(b'foo\nbar\nbaz\n')
    p = parser.LineBasedParser(s)
    self.assertEqual(b'foo', p.next_line())
    self.assertEqual(b'bar', p.next_line())
    p.push_line(b'bar')
    self.assertEqual(b'bar', p.next_line())
    self.assertEqual(b'baz', p.next_line())
    self.assertEqual(None, p.next_line())