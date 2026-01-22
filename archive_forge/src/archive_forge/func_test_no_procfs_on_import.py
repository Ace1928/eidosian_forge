from __future__ import division
import collections
import contextlib
import errno
import glob
import io
import os
import re
import shutil
import socket
import struct
import textwrap
import time
import unittest
import warnings
import psutil
from psutil import LINUX
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import basestring
from psutil._compat import u
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_GETLOADAVG
from psutil.tests import HAS_RLIMIT
from psutil.tests import PYPY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import ThreadTask
from psutil.tests import call_until
from psutil.tests import mock
from psutil.tests import reload_module
from psutil.tests import retry_on_failure
from psutil.tests import safe_rmpath
from psutil.tests import sh
from psutil.tests import skip_on_not_implemented
from psutil.tests import which
def test_no_procfs_on_import(self):
    my_procfs = self.get_testfn()
    os.mkdir(my_procfs)
    with open(os.path.join(my_procfs, 'stat'), 'w') as f:
        f.write('cpu   0 0 0 0 0 0 0 0 0 0\n')
        f.write('cpu0  0 0 0 0 0 0 0 0 0 0\n')
        f.write('cpu1  0 0 0 0 0 0 0 0 0 0\n')
    try:
        orig_open = open

        def open_mock(name, *args, **kwargs):
            if name.startswith('/proc'):
                raise IOError(errno.ENOENT, 'rejecting access for test')
            return orig_open(name, *args, **kwargs)
        patch_point = 'builtins.open' if PY3 else '__builtin__.open'
        with mock.patch(patch_point, side_effect=open_mock):
            reload_module(psutil)
            self.assertRaises(IOError, psutil.cpu_times)
            self.assertRaises(IOError, psutil.cpu_times, percpu=True)
            self.assertRaises(IOError, psutil.cpu_percent)
            self.assertRaises(IOError, psutil.cpu_percent, percpu=True)
            self.assertRaises(IOError, psutil.cpu_times_percent)
            self.assertRaises(IOError, psutil.cpu_times_percent, percpu=True)
            psutil.PROCFS_PATH = my_procfs
            self.assertEqual(psutil.cpu_percent(), 0)
            self.assertEqual(sum(psutil.cpu_times_percent()), 0)
            per_cpu_percent = psutil.cpu_percent(percpu=True)
            self.assertEqual(sum(per_cpu_percent), 0)
            per_cpu_times_percent = psutil.cpu_times_percent(percpu=True)
            self.assertEqual(sum(map(sum, per_cpu_times_percent)), 0)
            with open(os.path.join(my_procfs, 'stat'), 'w') as f:
                f.write('cpu   1 0 0 0 0 0 0 0 0 0\n')
                f.write('cpu0  1 0 0 0 0 0 0 0 0 0\n')
                f.write('cpu1  1 0 0 0 0 0 0 0 0 0\n')
            self.assertNotEqual(psutil.cpu_percent(), 0)
            self.assertNotEqual(sum(psutil.cpu_percent(percpu=True)), 0)
            self.assertNotEqual(sum(psutil.cpu_times_percent()), 0)
            self.assertNotEqual(sum(map(sum, psutil.cpu_times_percent(percpu=True))), 0)
    finally:
        shutil.rmtree(my_procfs)
        reload_module(psutil)
    self.assertEqual(psutil.PROCFS_PATH, '/proc')