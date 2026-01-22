import unittest
import unittest.mock
import queue as pyqueue
import textwrap
import time
import io
import itertools
import sys
import os
import gc
import errno
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import socket_helper
from test.support import threading_helper
from test.support import warnings_helper
import_helper.import_module('multiprocess.synchronize')
import threading
import multiprocess as multiprocessing
import multiprocess.connection
import multiprocess.dummy
import multiprocess.heap
import multiprocess.managers
import multiprocess.pool
import multiprocess.queues
from multiprocess import util
from multiprocess.connection import wait
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
@unittest.skipUnless(HAS_SHAREDCTYPES, 'needs sharedctypes')
def test_waitfor_timeout(self):
    cond = self.Condition()
    state = self.Value('i', 0)
    success = self.Value('i', False)
    sem = self.Semaphore(0)
    p = self.Process(target=self._test_waitfor_timeout_f, args=(cond, state, success, sem))
    p.daemon = True
    p.start()
    self.assertTrue(sem.acquire(timeout=support.LONG_TIMEOUT))
    for i in range(3):
        time.sleep(0.01)
        with cond:
            state.value += 1
            cond.notify()
    join_process(p)
    self.assertTrue(success.value)