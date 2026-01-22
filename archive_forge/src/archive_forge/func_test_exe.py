from __future__ import print_function
import functools
import os
import platform
import unittest
import psutil
import psutil._common
from psutil import LINUX
from psutil import MACOS
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import ProcessLookupError
from psutil._compat import super
from psutil.tests import HAS_CPU_AFFINITY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_IONICE
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_PROC_CPU_NUM
from psutil.tests import HAS_PROC_IO_COUNTERS
from psutil.tests import HAS_RLIMIT
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import TestMemoryLeak
from psutil.tests import create_sockets
from psutil.tests import get_testfn
from psutil.tests import process_namespace
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import system_namespace
from psutil.tests import terminate
@fewtimes_if_linux()
def test_exe(self):
    self.execute(self.proc.exe)