import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_auto_break_stale_lock_configured_off(self):
    """Automatic breaking can be turned off"""
    l1 = LockDir(self.get_transport(), 'a', extra_holder_info={'pid': '12312313'})
    config.GlobalStack().set('locks.steal_dead', False)
    token_1 = l1.attempt_lock()
    self.addCleanup(l1.unlock)
    l2 = LockDir(self.get_transport(), 'a')
    self.assertRaises(LockContention, l2.attempt_lock)
    l1.confirm()