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
def check_forkserver_death(self, signum):
    if self.TYPE == 'threads':
        self.skipTest('test not appropriate for {}'.format(self.TYPE))
    sm = multiprocessing.get_start_method()
    if sm != 'forkserver':
        self.skipTest('test not appropriate for {}'.format(sm))
    from multiprocess.forkserver import _forkserver
    _forkserver.ensure_running()
    delay = 0.5
    evt = self.Event()
    proc = self.Process(target=self._sleep_and_set_event, args=(evt, delay))
    proc.start()
    pid = _forkserver._forkserver_pid
    os.kill(pid, signum)
    time.sleep(delay * 2.0)
    evt2 = self.Event()
    proc2 = self.Process(target=self._sleep_and_set_event, args=(evt2,))
    proc2.start()
    proc2.join()
    self.assertTrue(evt2.is_set())
    self.assertEqual(proc2.exitcode, 0)
    proc.join()
    self.assertTrue(evt.is_set())
    self.assertIn(proc.exitcode, (0, 255))