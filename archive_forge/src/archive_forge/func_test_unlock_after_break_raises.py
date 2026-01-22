import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_unlock_after_break_raises(self):
    ld = self.get_lock()
    ld2 = self.get_lock()
    ld.create()
    ld.attempt_lock()
    ld2.force_break(ld2.peek())
    self.assertRaises(LockBroken, ld.unlock)