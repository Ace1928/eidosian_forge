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
def test_cpu_steal_decrease(self):
    content = textwrap.dedent('            cpu   0 0 0 0 0 0 0 1 0 0\n            cpu0  0 0 0 0 0 0 0 1 0 0\n            cpu1  0 0 0 0 0 0 0 1 0 0\n            ').encode()
    with mock_open_content({'/proc/stat': content}) as m:
        psutil.cpu_percent()
        assert m.called
        psutil.cpu_percent(percpu=True)
        psutil.cpu_times_percent()
        psutil.cpu_times_percent(percpu=True)
    content = textwrap.dedent('            cpu   1 0 0 0 0 0 0 0 0 0\n            cpu0  1 0 0 0 0 0 0 0 0 0\n            cpu1  1 0 0 0 0 0 0 0 0 0\n            ').encode()
    with mock_open_content({'/proc/stat': content}):
        cpu_percent = psutil.cpu_percent()
        assert m.called
        cpu_percent_percpu = psutil.cpu_percent(percpu=True)
        cpu_times_percent = psutil.cpu_times_percent()
        cpu_times_percent_percpu = psutil.cpu_times_percent(percpu=True)
        self.assertNotEqual(cpu_percent, 0)
        self.assertNotEqual(sum(cpu_percent_percpu), 0)
        self.assertNotEqual(sum(cpu_times_percent), 0)
        self.assertNotEqual(sum(cpu_times_percent), 100.0)
        self.assertNotEqual(sum(map(sum, cpu_times_percent_percpu)), 0)
        self.assertNotEqual(sum(map(sum, cpu_times_percent_percpu)), 100.0)
        self.assertEqual(cpu_times_percent.steal, 0)
        self.assertNotEqual(cpu_times_percent.user, 0)