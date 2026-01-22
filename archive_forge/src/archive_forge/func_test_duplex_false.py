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
def test_duplex_false(self):
    reader, writer = self.Pipe(duplex=False)
    self.assertEqual(writer.send(1), None)
    self.assertEqual(reader.recv(), 1)
    if self.TYPE == 'processes':
        self.assertEqual(reader.readable, True)
        self.assertEqual(reader.writable, False)
        self.assertEqual(writer.readable, False)
        self.assertEqual(writer.writable, True)
        self.assertRaises(OSError, reader.send, 2)
        self.assertRaises(OSError, writer.recv)
        self.assertRaises(OSError, writer.poll)