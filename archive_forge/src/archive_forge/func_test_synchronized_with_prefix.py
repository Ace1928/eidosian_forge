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
def test_synchronized_with_prefix(self):
    lock_name = 'mylock'
    lock_pfix = 'mypfix-'
    foo = lockutils.synchronized_with_prefix(lock_pfix)

    @foo(lock_name, external=True)
    def bar(dirpath, pfix, name):
        return True
    lock_dir = tempfile.mkdtemp()
    self.config(lock_path=lock_dir, group='oslo_concurrency')
    self.assertTrue(bar(lock_dir, lock_pfix, lock_name))