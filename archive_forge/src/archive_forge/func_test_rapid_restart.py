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
def test_rapid_restart(self):
    authkey = os.urandom(32)
    manager = QueueManager(address=(socket_helper.HOST, 0), authkey=authkey, serializer=SERIALIZER)
    try:
        srvr = manager.get_server()
        addr = srvr.address
        srvr.listener.close()
        manager.start()
        p = self.Process(target=self._putter, args=(manager.address, authkey))
        p.start()
        p.join()
        queue = manager.get_queue()
        self.assertEqual(queue.get(), 'hello world')
        del queue
    finally:
        if hasattr(manager, 'shutdown'):
            manager.shutdown()
    manager = QueueManager(address=addr, authkey=authkey, serializer=SERIALIZER)
    try:
        manager.start()
        self.addCleanup(manager.shutdown)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        time.sleep(1.0)
        manager = QueueManager(address=addr, authkey=authkey, serializer=SERIALIZER)
        if hasattr(manager, 'shutdown'):
            self.addCleanup(manager.shutdown)