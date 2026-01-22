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
def test_process__repr__(self, func=repr):
    p = psutil.Process(self.spawn_testproc().pid)
    r = func(p)
    self.assertIn('psutil.Process', r)
    self.assertIn('pid=%s' % p.pid, r)
    self.assertIn("name='%s'" % str(p.name()), r.replace("name=u'", "name='"))
    self.assertIn('status=', r)
    self.assertNotIn('exitcode=', r)
    p.terminate()
    p.wait()
    r = func(p)
    self.assertIn("status='terminated'", r)
    self.assertIn('exitcode=', r)
    with mock.patch.object(psutil.Process, 'name', side_effect=psutil.ZombieProcess(os.getpid())):
        p = psutil.Process()
        r = func(p)
        self.assertIn('pid=%s' % p.pid, r)
        self.assertIn("status='zombie'", r)
        self.assertNotIn('name=', r)
    with mock.patch.object(psutil.Process, 'name', side_effect=psutil.NoSuchProcess(os.getpid())):
        p = psutil.Process()
        r = func(p)
        self.assertIn('pid=%s' % p.pid, r)
        self.assertIn('terminated', r)
        self.assertNotIn('name=', r)
    with mock.patch.object(psutil.Process, 'name', side_effect=psutil.AccessDenied(os.getpid())):
        p = psutil.Process()
        r = func(p)
        self.assertIn('pid=%s' % p.pid, r)
        self.assertNotIn('name=', r)