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
def test_queue_feeder_on_queue_feeder_error(self):
    if self.TYPE != 'processes':
        self.skipTest('test not appropriate for {}'.format(self.TYPE))

    class NotSerializable(object):
        """Mock unserializable object"""

        def __init__(self):
            self.reduce_was_called = False
            self.on_queue_feeder_error_was_called = False

        def __reduce__(self):
            self.reduce_was_called = True
            raise AttributeError

    class SafeQueue(multiprocessing.queues.Queue):
        """Queue with overloaded _on_queue_feeder_error hook"""

        @staticmethod
        def _on_queue_feeder_error(e, obj):
            if isinstance(e, AttributeError) and isinstance(obj, NotSerializable):
                obj.on_queue_feeder_error_was_called = True
    not_serializable_obj = NotSerializable()
    with test.support.captured_stderr():
        q = SafeQueue(ctx=multiprocessing.get_context())
        q.put(not_serializable_obj)
        q.put(True)
        self.assertTrue(q.get(timeout=support.SHORT_TIMEOUT))
    self.assertTrue(not_serializable_obj.reduce_was_called)
    self.assertTrue(not_serializable_obj.on_queue_feeder_error_was_called)