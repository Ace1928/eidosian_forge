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
def test_shared_memory_cleaned_after_process_termination(self):
    cmd = "if 1:\n            import os, time, sys\n            from multiprocessing import shared_memory\n\n            # Create a shared_memory segment, and send the segment name\n            sm = shared_memory.SharedMemory(create=True, size=10)\n            sys.stdout.write(sm.name + '\\n')\n            sys.stdout.flush()\n            time.sleep(100)\n        "
    with subprocess.Popen([sys.executable, '-E', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        name = p.stdout.readline().strip().decode()
        p.terminate()
        p.wait()
        deadline = time.monotonic() + support.LONG_TIMEOUT
        t = 0.1
        while time.monotonic() < deadline:
            time.sleep(t)
            t = min(t * 2, 5)
            try:
                smm = shared_memory.SharedMemory(name, create=False)
            except FileNotFoundError:
                break
        else:
            raise AssertionError('A SharedMemory segment was leaked after a process was abruptly terminated.')
        if os.name == 'posix':
            resource_tracker.unregister(f'/{name}', 'shared_memory')
            err = p.stderr.read().decode()
            self.assertIn('resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown', err)