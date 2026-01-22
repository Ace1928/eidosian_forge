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
def test_emulate_unix(self):
    content = textwrap.dedent('            0: 00000003 000 000 0001 03 462170 @/tmp/dbus-Qw2hMPIU3n\n            0: 00000003 000 000 0001 03 35010 @/tmp/dbus-tB2X8h69BQ\n            0: 00000003 000 000 0001 03 34424 @/tmp/dbus-cHy80Y8O\n            000000000000000000000000000000000000000000000000000000\n            ')
    with mock_open_content({'/proc/net/unix': content}) as m:
        psutil.net_connections(kind='unix')
        assert m.called