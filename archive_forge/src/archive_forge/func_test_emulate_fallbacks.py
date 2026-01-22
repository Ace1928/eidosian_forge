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
def test_emulate_fallbacks(self):
    import psutil._pslinux
    original = psutil._pslinux.cpu_count_logical()
    with mock.patch('psutil._pslinux.os.sysconf', side_effect=ValueError) as m:
        self.assertEqual(psutil._pslinux.cpu_count_logical(), original)
        assert m.called
        with mock.patch('psutil._common.open', create=True) as m:
            self.assertIsNone(psutil._pslinux.cpu_count_logical())
            self.assertEqual(m.call_count, 2)
            self.assertEqual(m.call_args[0][0], '/proc/stat')
        with open('/proc/cpuinfo', 'rb') as f:
            cpuinfo_data = f.read()
        fake_file = io.BytesIO(cpuinfo_data)
        with mock.patch('psutil._common.open', return_value=fake_file, create=True) as m:
            self.assertEqual(psutil._pslinux.cpu_count_logical(), original)
        with mock_open_content({'/proc/cpuinfo': b''}) as m:
            self.assertEqual(psutil._pslinux.cpu_count_logical(), original)
            assert m.called