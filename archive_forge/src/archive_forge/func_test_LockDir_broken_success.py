import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_LockDir_broken_success(self):
    ld = self.get_lock()
    ld.create()
    ld2 = self.get_lock()
    result = ld.attempt_lock()
    LockDir.hooks.install_named_hook('lock_broken', self.record_hook, 'record_hook')
    ld2.force_break(ld2.peek())
    lock_path = ld.transport.abspath(ld.path)
    self.assertEqual([lock.LockResult(lock_path, result)], self._calls)