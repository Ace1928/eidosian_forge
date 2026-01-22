import io
import time
import unittest
from fastimport import (
from :2
def test_done_without_feature(self):
    s = io.BytesIO(b'done\nmore data\n')
    p = parser.ImportParser(s)
    cmds = p.iter_commands()
    self.assertEqual([], list(cmds))