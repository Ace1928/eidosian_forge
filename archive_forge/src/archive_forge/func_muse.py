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
def muse(field):
    """Thin wrapper around 'muse' cmdline utility."""
    out = sh('muse')
    for line in out.split('\n'):
        if line.startswith(field):
            break
    else:
        raise ValueError('line not found')
    return int(line.split()[1])