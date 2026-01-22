import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_lock_holder_live_process(self):
    """Detect that the holder (this process) is still running."""
    info = LockHeldInfo.for_this_process(None)
    self.assertFalse(info.is_lock_holder_known_dead())