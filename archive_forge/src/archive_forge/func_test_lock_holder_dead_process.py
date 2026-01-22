import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_lock_holder_dead_process(self):
    """Detect that the holder (this process) is still running."""
    self.overrideAttr(lockdir, 'get_host_name', lambda: 'aproperhostname')
    info = LockHeldInfo.for_this_process(None)
    info.info_dict['pid'] = '123123123'
    self.assertTrue(info.is_lock_holder_known_dead())