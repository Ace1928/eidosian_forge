import doctest
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from zope.testing import setupstack
import zc.lockfile
def test_unlock_and_lock_while_multiprocessing_process_running(self):
    import multiprocessing
    lock = zc.lockfile.LockFile('l')
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=q.get)
    p.daemon = True
    p.start()
    lock.close()
    lock = zc.lockfile.LockFile('l')
    self.assertTrue(p.is_alive())
    q.put(0)
    lock.close()
    p.join()