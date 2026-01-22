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
def test_parse_smaps_mocked(self):
    content = textwrap.dedent('            fffff0 r-xp 00000000 00:00 0                  [vsyscall]\n            Size:                  1 kB\n            Rss:                   2 kB\n            Pss:                   3 kB\n            Shared_Clean:          4 kB\n            Shared_Dirty:          5 kB\n            Private_Clean:         6 kB\n            Private_Dirty:         7 kB\n            Referenced:            8 kB\n            Anonymous:             9 kB\n            LazyFree:              10 kB\n            AnonHugePages:         11 kB\n            ShmemPmdMapped:        12 kB\n            Shared_Hugetlb:        13 kB\n            Private_Hugetlb:       14 kB\n            Swap:                  15 kB\n            SwapPss:               16 kB\n            KernelPageSize:        17 kB\n            MMUPageSize:           18 kB\n            Locked:                19 kB\n            VmFlags: rd ex\n            ').encode()
    with mock_open_content({'/proc/%s/smaps' % os.getpid(): content}) as m:
        p = psutil._pslinux.Process(os.getpid())
        uss, pss, swap = p._parse_smaps()
        assert m.called
        self.assertEqual(uss, (6 + 7 + 14) * 1024)
        self.assertEqual(pss, 3 * 1024)
        self.assertEqual(swap, 15 * 1024)