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
def test_vmem_free(self):
    vmstat_val = vm_stat('free')
    psutil_val = psutil.virtual_memory().free
    self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)