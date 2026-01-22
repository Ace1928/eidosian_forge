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
def test_warnings_on_misses(self):
    content = textwrap.dedent('            Active(anon):    6145416 kB\n            Active(file):    2950064 kB\n            Inactive(anon):   574764 kB\n            Inactive(file):  1567648 kB\n            MemAvailable:         -1 kB\n            MemFree:         2057400 kB\n            MemTotal:       16325648 kB\n            SReclaimable:     346648 kB\n            ').encode()
    with mock_open_content({'/proc/meminfo': content}) as m:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            ret = psutil.virtual_memory()
            assert m.called
            self.assertEqual(len(ws), 1)
            w = ws[0]
            self.assertIn("memory stats couldn't be determined", str(w.message))
            self.assertIn('cached', str(w.message))
            self.assertIn('shared', str(w.message))
            self.assertIn('active', str(w.message))
            self.assertIn('inactive', str(w.message))
            self.assertIn('buffers', str(w.message))
            self.assertIn('available', str(w.message))
            self.assertEqual(ret.cached, 0)
            self.assertEqual(ret.active, 0)
            self.assertEqual(ret.inactive, 0)
            self.assertEqual(ret.shared, 0)
            self.assertEqual(ret.buffers, 0)
            self.assertEqual(ret.available, 0)
            self.assertEqual(ret.slab, 0)