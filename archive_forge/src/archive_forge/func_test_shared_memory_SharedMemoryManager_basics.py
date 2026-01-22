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
def test_shared_memory_SharedMemoryManager_basics(self):
    smm1 = multiprocessing.managers.SharedMemoryManager()
    with self.assertRaises(ValueError):
        smm1.SharedMemory(size=9)
    smm1.start()
    lol = [smm1.ShareableList(range(i)) for i in range(5, 10)]
    lom = [smm1.SharedMemory(size=j) for j in range(32, 128, 16)]
    doppleganger_list0 = shared_memory.ShareableList(name=lol[0].shm.name)
    self.assertEqual(len(doppleganger_list0), 5)
    doppleganger_shm0 = shared_memory.SharedMemory(name=lom[0].name)
    self.assertGreaterEqual(len(doppleganger_shm0.buf), 32)
    held_name = lom[0].name
    smm1.shutdown()
    if sys.platform != 'win32':
        with self.assertRaises(FileNotFoundError):
            absent_shm = shared_memory.SharedMemory(name=held_name)
    with multiprocessing.managers.SharedMemoryManager() as smm2:
        sl = smm2.ShareableList('howdy')
        shm = smm2.SharedMemory(size=128)
        held_name = sl.shm.name
    if sys.platform != 'win32':
        with self.assertRaises(FileNotFoundError):
            absent_sl = shared_memory.ShareableList(name=held_name)