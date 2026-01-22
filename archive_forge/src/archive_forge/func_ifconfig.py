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
def ifconfig(nic):
    ret = {}
    out = sh('ifconfig %s' % nic)
    ret['packets_recv'] = int(re.findall('RX packets[: ](\\d+)', out)[0])
    ret['packets_sent'] = int(re.findall('TX packets[: ](\\d+)', out)[0])
    ret['errin'] = int(re.findall('errors[: ](\\d+)', out)[0])
    ret['errout'] = int(re.findall('errors[: ](\\d+)', out)[1])
    ret['dropin'] = int(re.findall('dropped[: ](\\d+)', out)[0])
    ret['dropout'] = int(re.findall('dropped[: ](\\d+)', out)[1])
    ret['bytes_recv'] = int(re.findall('RX (?:packets \\d+ +)?bytes[: ](\\d+)', out)[0])
    ret['bytes_sent'] = int(re.findall('TX (?:packets \\d+ +)?bytes[: ](\\d+)', out)[0])
    return ret