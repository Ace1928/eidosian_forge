import platform
import re
import time
import unittest
import psutil
from psutil import MACOS
from psutil import POSIX
from psutil.tests import HAS_BATTERY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate
@retry_on_failure()
def test_disks(self):

    def df(path):
        out = sh('df -k "%s"' % path).strip()
        lines = out.split('\n')
        lines.pop(0)
        line = lines.pop(0)
        dev, total, used, free = line.split()[:4]
        if dev == 'none':
            dev = ''
        total = int(total) * 1024
        used = int(used) * 1024
        free = int(free) * 1024
        return (dev, total, used, free)
    for part in psutil.disk_partitions(all=False):
        usage = psutil.disk_usage(part.mountpoint)
        dev, total, used, free = df(part.mountpoint)
        self.assertEqual(part.device, dev)
        self.assertEqual(usage.total, total)
        self.assertAlmostEqual(usage.free, free, delta=TOLERANCE_DISK_USAGE)
        self.assertAlmostEqual(usage.used, used, delta=TOLERANCE_DISK_USAGE)