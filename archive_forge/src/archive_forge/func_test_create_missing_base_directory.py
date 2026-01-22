import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_create_missing_base_directory(self):
    """If LockDir.path doesn't exist, it can be created

        Some people manually remove the entire lock/ directory trying
        to unlock a stuck repository/branch/etc. Rather than failing
        after that, just create the lock directory when needed.
        """
    t = self.get_transport()
    lf1 = LockDir(t, 'test_lock')
    lf1.create()
    self.assertTrue(t.has('test_lock'))
    t.rmdir('test_lock')
    self.assertFalse(t.has('test_lock'))
    lf1.lock_write()
    self.assertTrue(t.has('test_lock'))
    self.assertTrue(t.has('test_lock/held/info'))
    lf1.unlock()
    self.assertFalse(t.has('test_lock/held/info'))