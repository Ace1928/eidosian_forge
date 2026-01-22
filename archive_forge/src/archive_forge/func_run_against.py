import ast
import collections
import errno
import json
import os
import pickle
import socket
import stat
import unittest
import psutil
import psutil.tests
from psutil import LINUX
from psutil import POSIX
from psutil import WINDOWS
from psutil._common import bcat
from psutil._common import cat
from psutil._common import debug
from psutil._common import isfile_strict
from psutil._common import memoize
from psutil._common import memoize_when_activated
from psutil._common import parse_environ_block
from psutil._common import supports_ipv6
from psutil._common import wrap_numbers
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import redirect_stderr
from psutil.tests import APPVEYOR
from psutil.tests import CI_TESTING
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import SCRIPTS_DIR
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import reload_module
from psutil.tests import sh
def run_against(self, obj, expected_retval=None):
    for _ in range(2):
        ret = obj()
        self.assertEqual(self.calls, [((), {})])
        if expected_retval is not None:
            self.assertEqual(ret, expected_retval)
    for _ in range(2):
        ret = obj(1)
        self.assertEqual(self.calls, [((), {}), ((1,), {})])
        if expected_retval is not None:
            self.assertEqual(ret, expected_retval)
    for _ in range(2):
        ret = obj(1, bar=2)
        self.assertEqual(self.calls, [((), {}), ((1,), {}), ((1,), {'bar': 2})])
        if expected_retval is not None:
            self.assertEqual(ret, expected_retval)
    self.assertEqual(len(self.calls), 3)
    obj.cache_clear()
    ret = obj()
    if expected_retval is not None:
        self.assertEqual(ret, expected_retval)
    self.assertEqual(len(self.calls), 4)
    self.assertEqual(obj.__doc__, 'My docstring.')