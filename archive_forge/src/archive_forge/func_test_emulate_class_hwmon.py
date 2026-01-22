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
def test_emulate_class_hwmon(self):

    def open_mock(name, *args, **kwargs):
        if name.endswith('/name'):
            return io.StringIO(u('name'))
        elif name.endswith('/temp1_label'):
            return io.StringIO(u('label'))
        elif name.endswith('/temp1_input'):
            return io.BytesIO(b'30000')
        elif name.endswith('/temp1_max'):
            return io.BytesIO(b'40000')
        elif name.endswith('/temp1_crit'):
            return io.BytesIO(b'50000')
        else:
            return orig_open(name, *args, **kwargs)
    orig_open = open
    patch_point = 'builtins.open' if PY3 else '__builtin__.open'
    with mock.patch(patch_point, side_effect=open_mock):
        with mock.patch('glob.glob', return_value=['/sys/class/hwmon/hwmon0/temp1']):
            temp = psutil.sensors_temperatures()['name'][0]
            self.assertEqual(temp.label, 'label')
            self.assertEqual(temp.current, 30.0)
            self.assertEqual(temp.high, 40.0)
            self.assertEqual(temp.critical, 50.0)