import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_LockDir_broken_failure(self):
    ld = self.get_lock()
    ld.create()
    ld2 = self.get_lock()
    result = ld.attempt_lock()
    holder_info = ld2.peek()
    ld.unlock()
    LockDir.hooks.install_named_hook('lock_broken', self.record_hook, 'record_hook')
    ld2.force_break(holder_info)
    lock_path = ld.transport.abspath(ld.path)
    self.assertEqual([], self._calls)