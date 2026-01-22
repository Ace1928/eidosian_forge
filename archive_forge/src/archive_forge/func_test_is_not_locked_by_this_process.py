import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_is_not_locked_by_this_process(self):
    info = LockHeldInfo.for_this_process(None)
    info.info_dict['pid'] = '123123123123123'
    self.assertFalse(info.is_locked_by_this_process())