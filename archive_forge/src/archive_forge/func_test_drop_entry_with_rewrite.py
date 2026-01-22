from io import BytesIO
from dulwich.tests import TestCase
from ..objects import ZERO_SHA
from ..reflog import (
def test_drop_entry_with_rewrite(self):
    drop_reflog_entry(self.f, 1, True)
    log = self._read_log()
    self.assertEqual(len(log), 2)
    self.assertEqual(self.original_log[0], log[0])
    self.assertEqual(self.original_log[0].new_sha, log[1].old_sha)
    self.assertEqual(self.original_log[2].new_sha, log[1].new_sha)
    self.f.seek(0)
    drop_reflog_entry(self.f, 1, True)
    log = self._read_log()
    self.assertEqual(len(log), 1)
    self.assertEqual(ZERO_SHA, log[0].old_sha)
    self.assertEqual(self.original_log[2].new_sha, log[0].new_sha)