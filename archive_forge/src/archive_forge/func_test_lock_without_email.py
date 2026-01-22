import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_lock_without_email(self):
    global_config = config.GlobalStack()
    global_config.set('email', 'User Identity')
    ld1 = self.get_lock()
    ld1.create()
    ld1.lock_write()
    ld1.unlock()