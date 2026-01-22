import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_failed_lock_leaves_no_trash(self):
    ld1 = self.get_lock()
    ld2 = self.get_lock()
    ld1.create()
    t = self.get_transport().clone('test_lock')

    def check_dir(a):
        self.assertEqual(a, t.list_dir('.'))
    check_dir([])
    ld1.attempt_lock()
    self.addCleanup(ld1.unlock)
    check_dir(['held'])
    self.assertRaises(errors.LockContention, ld2.attempt_lock)
    check_dir(['held'])