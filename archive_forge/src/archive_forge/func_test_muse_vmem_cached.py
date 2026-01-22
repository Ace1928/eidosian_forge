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
@unittest.skipIf(not MUSE_AVAILABLE, 'muse not installed')
@retry_on_failure()
def test_muse_vmem_cached(self):
    num = muse('Cache')
    self.assertAlmostEqual(psutil.virtual_memory().cached, num, delta=TOLERANCE_SYS_MEM)