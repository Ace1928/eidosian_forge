import datetime
import os
import re
import time
import unittest
import psutil
from psutil import BSD
from psutil import FREEBSD
from psutil import NETBSD
from psutil import OPENBSD
from psutil.tests import HAS_BATTERY
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate
from psutil.tests import which
def test_cpu_stats_interrupts(self):
    with open('/proc/stat', 'rb') as f:
        for line in f:
            if line.startswith(b'intr'):
                interrupts = int(line.split()[1])
                break
        else:
            raise ValueError("couldn't find line")
    self.assertAlmostEqual(psutil.cpu_stats().interrupts, interrupts, delta=1000)