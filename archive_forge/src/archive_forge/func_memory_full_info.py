import enum
import errno
import multiprocessing
import os
import stat
import time
import traceback
import psutil
from psutil import AIX
from psutil import BSD
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import OSX
from psutil import POSIX
from psutil import WINDOWS
from psutil._compat import PY3
from psutil._compat import long
from psutil._compat import unicode
from psutil.tests import CI_TESTING
from psutil.tests import VALID_PROC_STATUSES
from psutil.tests import PsutilTestCase
from psutil.tests import check_connection_ntuple
from psutil.tests import create_sockets
from psutil.tests import is_namedtuple
from psutil.tests import is_win_secure_system_proc
from psutil.tests import process_namespace
from psutil.tests import serialrun
def memory_full_info(self, ret, info):
    assert is_namedtuple(ret)
    total = psutil.virtual_memory().total
    for name in ret._fields:
        value = getattr(ret, name)
        self.assertIsInstance(value, (int, long))
        self.assertGreaterEqual(value, 0, msg=(name, value))
        if LINUX or (OSX and name in ('vms', 'data')):
            continue
        self.assertLessEqual(value, total, msg=(name, value, total))
    if LINUX:
        self.assertGreaterEqual(ret.pss, ret.uss)