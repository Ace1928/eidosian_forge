import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_10_lock_uncontested(self):
    """Acquire and release a lock"""
    t = self.get_transport()
    lf = LockDir(t, 'test_lock')
    lf.create()
    lf.attempt_lock()
    try:
        self.assertTrue(lf.is_held)
    finally:
        lf.unlock()
        self.assertFalse(lf.is_held)