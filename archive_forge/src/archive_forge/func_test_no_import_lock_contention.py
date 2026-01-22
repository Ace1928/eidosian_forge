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
def test_no_import_lock_contention(self):
    with os_helper.temp_cwd():
        module_name = 'imported_by_an_imported_module'
        with open(module_name + '.py', 'w', encoding='utf-8') as f:
            f.write("if 1:\n                    import multiprocess as multiprocessing\n\n                    q = multiprocessing.Queue()\n                    q.put('knock knock')\n                    q.get(timeout=3)\n                    q.close()\n                    del q\n                ")
        with import_helper.DirsOnSysPath(os.getcwd()):
            try:
                __import__(module_name)
            except pyqueue.Empty:
                self.fail('Probable regression on import lock contention; see Issue #22853')