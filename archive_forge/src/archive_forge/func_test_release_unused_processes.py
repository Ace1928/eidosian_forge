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
def test_release_unused_processes(self):
    will_fail_in = 3
    forked_processes = []

    class FailingForkProcess:

        def __init__(self, **kwargs):
            self.name = 'Fake Process'
            self.exitcode = None
            self.state = None
            forked_processes.append(self)

        def start(self):
            nonlocal will_fail_in
            if will_fail_in <= 0:
                raise OSError('Manually induced OSError')
            will_fail_in -= 1
            self.state = 'started'

        def terminate(self):
            self.state = 'stopping'

        def join(self):
            if self.state == 'stopping':
                self.state = 'stopped'

        def is_alive(self):
            return self.state == 'started' or self.state == 'stopping'
    with self.assertRaisesRegex(OSError, 'Manually induced OSError'):
        p = multiprocessing.pool.Pool(5, context=unittest.mock.MagicMock(Process=FailingForkProcess))
        p.close()
        p.join()
    self.assertFalse(any((process.is_alive() for process in forked_processes)))