import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_LockDir_released_fail(self):
    ld = self.get_lock()
    ld.create()
    ld2 = self.get_lock()
    ld.attempt_lock()
    ld2.force_break(ld2.peek())
    LockDir.hooks.install_named_hook('lock_released', self.record_hook, 'record_hook')
    self.assertRaises(LockBroken, ld.unlock)
    self.assertEqual([], self._calls)