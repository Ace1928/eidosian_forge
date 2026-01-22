import os
import sys
import time
import shutil
import platform
import tempfile
import unittest
import multiprocessing
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
@unittest.skipIf(platform.system().lower() == 'windows', 'Unsupported on Windows')
def test_lock_local_storage(self):
    lock = LockLocalStorage('/tmp/a')
    with lock:
        self.assertTrue(True)
    lock = LockLocalStorage('/tmp/b', timeout=0.5)
    with lock:
        expected_msg = 'Failed to acquire thread lock'
        self.assertRaisesRegex(LibcloudError, expected_msg, lock.__enter__)
    success_1 = multiprocessing.Value('i', 0)
    success_2 = multiprocessing.Value('i', 0)
    p1 = multiprocessing.Process(target=PickleableAcquireLockInSubprocess(), args=(1, success_1))
    p1.start()
    time.sleep(0.2)
    p2 = multiprocessing.Process(target=PickleableAcquireLockInSubprocess(), args=(2, success_2))
    p2.start()
    p1.join()
    p2.join()
    self.assertEqual(bool(success_1.value), True, "Check didn't pass")
    self.assertEqual(bool(success_2.value), True, "Second check didn't pass")