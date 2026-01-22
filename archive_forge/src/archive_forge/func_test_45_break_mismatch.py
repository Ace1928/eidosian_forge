import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_45_break_mismatch(self):
    """Lock break races with someone else acquiring it"""
    t = self.get_transport()
    lf1 = LockDir(t, 'test_lock')
    lf1.create()
    lf1.attempt_lock()
    lf2 = LockDir(t, 'test_lock')
    holder_info = lf2.peek()
    lf1.unlock()
    lf3 = LockDir(t, 'test_lock')
    lf3.attempt_lock()
    self.assertRaises(LockBreakMismatch, lf2.force_break, holder_info)
    lf3.unlock()