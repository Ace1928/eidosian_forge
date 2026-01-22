import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_01_lock_repr(self):
    """Lock string representation"""
    lf = LockDir(self.get_transport(), 'test_lock')
    r = repr(lf)
    self.assertContainsRe(r, '^LockDir\\(.*/test_lock\\)$')