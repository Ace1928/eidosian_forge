import collections
import errno
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
from oslotest import base as test_base
from oslo_concurrency.fixture import lockutils as fixtures
from oslo_concurrency import lockutils
from oslo_config import fixture as config
def test_contextlock(self):
    self.config(lock_path=tempfile.mkdtemp(), group='oslo_concurrency')
    with lockutils.lock('test') as sem:
        self.assertIsInstance(sem, threading.Semaphore)
        with lockutils.lock('test2', external=True) as lock:
            self.assertTrue(lock.exists())
        with lockutils.lock('test1', external=True) as lock1:
            self.assertIsInstance(lock1, lockutils.InterProcessLock)