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
@unittest.skipIf(True, 'ModuleNotFoundError')
@only_run_in_spawn_testsuite('avoids redundant testing.')
def test_spawn_sys_executable_none_allows_import(self):
    rc, out, err = script_helper.assert_python_ok('-c', 'if 1:\n            import sys\n            sys.executable = None\n            assert "multiprocess" not in sys.modules, "already imported!"\n            import multiprocess as multiprocessing\n            import multiprocess.spawn  # This should not fail\n')
    self.assertEqual(rc, 0)
    self.assertFalse(err, msg=err.decode('utf-8'))