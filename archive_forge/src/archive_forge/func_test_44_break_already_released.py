import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_44_break_already_released(self):
    """Lock break races with regular release"""
    t = self.get_transport()
    lf1 = LockDir(t, 'test_lock')
    lf1.create()
    lf1.attempt_lock()
    lf2 = LockDir(t, 'test_lock')
    holder_info = lf2.peek()
    lf1.unlock()
    lf2.force_break(holder_info)
    lf2.attempt_lock()
    self.addCleanup(lf2.unlock)
    lf2.confirm()