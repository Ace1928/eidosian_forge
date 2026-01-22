import datetime
import errno
import glob
import os
import platform
import re
import signal
import subprocess
import sys
import time
import unittest
import warnings
import psutil
from psutil import WINDOWS
from psutil._compat import FileNotFoundError
from psutil._compat import super
from psutil._compat import which
from psutil.tests import APPVEYOR
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import HAS_BATTERY
from psutil.tests import IS_64BIT
from psutil.tests import PY3
from psutil.tests import PYPY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate
def test_boot_time_fluctuation(self):
    with mock.patch('psutil._pswindows.cext.boot_time', return_value=5):
        self.assertEqual(psutil.boot_time(), 5)
    with mock.patch('psutil._pswindows.cext.boot_time', return_value=4):
        self.assertEqual(psutil.boot_time(), 5)
    with mock.patch('psutil._pswindows.cext.boot_time', return_value=6):
        self.assertEqual(psutil.boot_time(), 5)
    with mock.patch('psutil._pswindows.cext.boot_time', return_value=333):
        self.assertEqual(psutil.boot_time(), 333)