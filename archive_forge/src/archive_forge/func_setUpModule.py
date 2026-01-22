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
def setUpModule():
    multiprocessing.set_forkserver_preload(PRELOAD)
    multiprocessing.process._cleanup()
    dangling[0] = multiprocessing.process._dangling.copy()
    dangling[1] = threading._dangling.copy()
    old_start_method[0] = multiprocessing.get_start_method(allow_none=True)
    try:
        multiprocessing.set_start_method(start_method, force=True)
    except ValueError:
        raise unittest.SkipTest(start_method + ' start method not supported')
    if sys.platform.startswith('linux'):
        try:
            lock = multiprocessing.RLock()
        except OSError:
            raise unittest.SkipTest('OSError raises on RLock creation, see issue 3111!')
    check_enough_semaphores()
    util.get_temp_dir()
    multiprocessing.get_logger().setLevel(LOG_LEVEL)