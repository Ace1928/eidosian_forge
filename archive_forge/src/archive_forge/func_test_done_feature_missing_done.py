import io
import time
import unittest
from fastimport import (
from :2
def test_done_feature_missing_done(self):
    s = io.BytesIO(b'feature done\n')
    p = parser.ImportParser(s)
    cmds = p.iter_commands()
    self.assertEqual(b'feature', next(cmds).name)
    self.assertRaises(errors.PrematureEndOfStream, lambda: next(cmds))