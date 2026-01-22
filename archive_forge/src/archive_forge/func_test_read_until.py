import io
import time
import unittest
from fastimport import (
from :2
def test_read_until(self):
    return
    s = io.BytesIO(b'foo\nbar\nbaz\nabc\ndef\nghi\n')
    p = parser.LineBasedParser(s)
    self.assertEqual(b'foo\nbar', p.read_until(b'baz'))
    self.assertEqual(b'abc', p.next_line())
    p.push_line(b'abc')
    self.assertEqual(b'def', p.read_until(b'ghi'))
    self.assertRaises(errors.MissingTerminator, p.read_until(b'>>>'))