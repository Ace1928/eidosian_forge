import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_12_lock_readonly_transport(self):
    """Fail to lock on readonly transport"""
    lf = LockDir(self.get_transport(), 'test_lock')
    lf.create()
    lf = LockDir(self.get_readonly_transport(), 'test_lock')
    self.assertRaises(LockFailed, lf.attempt_lock)