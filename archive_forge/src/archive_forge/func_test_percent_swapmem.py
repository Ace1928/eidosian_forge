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
def test_percent_swapmem(self):
    if psutil.swap_memory().total > 0:
        w = wmi.WMI().Win32_PerfRawData_PerfOS_PagingFile(Name='_Total')[0]
        percentSwap = int(w.PercentUsage) * 100 / int(w.PercentUsage_Base)
        self.assertGreaterEqual(psutil.swap_memory().percent, 0)
        self.assertAlmostEqual(psutil.swap_memory().percent, percentSwap, delta=5)
        self.assertLessEqual(psutil.swap_memory().percent, 100)