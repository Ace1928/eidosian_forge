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
import functools
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pathlib
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import script_helper
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
from multiprocess.connection import wait, AuthenticationError
from multiprocess import util
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
@support.requires_resource('cpu')
def test_args_argument(self):
    args_cases = (1, 'str', [1], (1,))
    args_types = (list, tuple)
    test_cases = itertools.product(args_cases, args_types)
    for args, args_type in test_cases:
        with self.subTest(args=args, args_type=args_type):
            q = self.Queue(1)
            p = self.Process(target=self._test_args, args=args_type((q, args)))
            p.daemon = True
            p.start()
            child_args = q.get()
            self.assertEqual(child_args, args)
            p.join()
            close_queue(q)