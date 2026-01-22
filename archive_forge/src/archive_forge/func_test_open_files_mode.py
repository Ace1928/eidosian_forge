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
@unittest.skipIf(PYPY, 'unreliable on PYPY')
def test_open_files_mode(self):

    def get_test_file(fname):
        p = psutil.Process()
        giveup_at = time.time() + GLOBAL_TIMEOUT
        while True:
            for file in p.open_files():
                if file.path == os.path.abspath(fname):
                    return file
                elif time.time() > giveup_at:
                    break
        raise RuntimeError('timeout looking for test file')
    testfn = self.get_testfn()
    with open(testfn, 'w'):
        self.assertEqual(get_test_file(testfn).mode, 'w')
    with open(testfn):
        self.assertEqual(get_test_file(testfn).mode, 'r')
    with open(testfn, 'a'):
        self.assertEqual(get_test_file(testfn).mode, 'a')
    with open(testfn, 'r+'):
        self.assertEqual(get_test_file(testfn).mode, 'r+')
    with open(testfn, 'w+'):
        self.assertEqual(get_test_file(testfn).mode, 'r+')
    with open(testfn, 'a+'):
        self.assertEqual(get_test_file(testfn).mode, 'a+')
    if PY3:
        safe_rmpath(testfn)
        with open(testfn, 'x'):
            self.assertEqual(get_test_file(testfn).mode, 'w')
        safe_rmpath(testfn)
        with open(testfn, 'x+'):
            self.assertEqual(get_test_file(testfn).mode, 'r+')