import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_LockDir_acquired_fail(self):
    ld = self.get_lock()
    ld.create()
    ld2 = self.get_lock()
    ld2.attempt_lock()
    LockDir.hooks.install_named_hook('lock_acquired', self.record_hook, 'record_hook')
    self.assertRaises(errors.LockContention, ld.attempt_lock)
    self.assertEqual([], self._calls)
    ld2.unlock()
    self.assertEqual([], self._calls)