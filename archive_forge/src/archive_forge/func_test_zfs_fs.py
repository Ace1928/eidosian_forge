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
def test_zfs_fs(self):
    with open('/proc/filesystems') as f:
        data = f.read()
    if 'zfs' in data:
        for part in psutil.disk_partitions():
            if part.fstype == 'zfs':
                break
        else:
            raise self.fail("couldn't find any ZFS partition")
    else:
        fake_file = io.StringIO(u('nodev\tzfs\n'))
        with mock.patch('psutil._common.open', return_value=fake_file, create=True) as m1:
            with mock.patch('psutil._pslinux.cext.disk_partitions', return_value=[('/dev/sdb3', '/', 'zfs', 'rw')]) as m2:
                ret = psutil.disk_partitions()
                assert m1.called
                assert m2.called
                assert ret
                self.assertEqual(ret[0].fstype, 'zfs')