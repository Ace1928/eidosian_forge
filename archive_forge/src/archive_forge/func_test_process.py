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
def test_process(self):
    q = self.Queue(1)
    e = self.Event()
    args = (q, 1, 2)
    kwargs = {'hello': 23, 'bye': 2.54}
    name = 'SomeProcess'
    p = self.Process(target=self._test, args=args, kwargs=kwargs, name=name)
    p.daemon = True
    current = self.current_process()
    if self.TYPE != 'threads':
        self.assertEqual(p.authkey, current.authkey)
    self.assertEqual(p.is_alive(), False)
    self.assertEqual(p.daemon, True)
    self.assertNotIn(p, self.active_children())
    self.assertTrue(type(self.active_children()) is list)
    self.assertEqual(p.exitcode, None)
    p.start()
    self.assertEqual(p.exitcode, None)
    self.assertEqual(p.is_alive(), True)
    self.assertIn(p, self.active_children())
    self.assertEqual(q.get(), args[1:])
    self.assertEqual(q.get(), kwargs)
    self.assertEqual(q.get(), p.name)
    if self.TYPE != 'threads':
        self.assertEqual(q.get(), current.authkey)
        self.assertEqual(q.get(), p.pid)
    p.join()
    self.assertEqual(p.exitcode, 0)
    self.assertEqual(p.is_alive(), False)
    self.assertNotIn(p, self.active_children())
    close_queue(q)