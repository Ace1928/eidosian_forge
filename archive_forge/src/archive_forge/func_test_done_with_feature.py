import io
import time
import unittest
from fastimport import (
from :2
def test_done_with_feature(self):
    s = io.BytesIO(b'feature done\ndone\nmore data\n')
    p = parser.ImportParser(s)
    cmds = p.iter_commands()
    self.assertEqual(b'feature', next(cmds).name)
    self.assertRaises(StopIteration, lambda: next(cmds))