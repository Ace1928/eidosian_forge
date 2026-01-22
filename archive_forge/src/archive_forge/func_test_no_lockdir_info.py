import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_no_lockdir_info(self):
    """We can cope with empty info files."""
    t = self.get_transport()
    t.mkdir('test_lock')
    t.mkdir('test_lock/held')
    t.put_bytes('test_lock/held/info', b'')
    lf = LockDir(t, 'test_lock')
    info = lf.peek()
    formatted_info = info.to_readable_dict()
    self.assertEqual(dict(user='<unknown>', hostname='<unknown>', pid='<unknown>', time_ago='(unknown)'), formatted_info)