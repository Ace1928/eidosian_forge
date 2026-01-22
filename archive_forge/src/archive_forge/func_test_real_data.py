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
def test_real_data(self):
    d = {'nvme0n1': (300, 508, 640, 1571, 5970, 1987, 2049, 451751, 47048), 'nvme0n1p1': (1171, 2, 5600256, 1024, 516, 0, 0, 0, 8), 'nvme0n1p2': (54, 54, 2396160, 5165056, 4, 24, 30, 1207, 28), 'nvme0n1p3': (2389, 4539, 5154, 150, 4828, 1844, 2019, 398, 348)}
    self.assertEqual(wrap_numbers(d, 'disk_io'), d)
    self.assertEqual(wrap_numbers(d, 'disk_io'), d)
    d = {'nvme0n1': (100, 508, 640, 1571, 5970, 1987, 2049, 451751, 47048), 'nvme0n1p1': (1171, 2, 5600256, 1024, 516, 0, 0, 0, 8), 'nvme0n1p2': (54, 54, 2396160, 5165056, 4, 24, 30, 1207, 28), 'nvme0n1p3': (2389, 4539, 5154, 150, 4828, 1844, 2019, 398, 348)}
    out = wrap_numbers(d, 'disk_io')
    self.assertEqual(out['nvme0n1'][0], 400)