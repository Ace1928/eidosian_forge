import contextlib
import datetime
import errno
import os
import platform
import pprint
import shutil
import signal
import socket
import sys
import time
import unittest
import psutil
from psutil import AIX
from psutil import BSD
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import long
from psutil.tests import ASCII_FS
from psutil.tests import CI_TESTING
from psutil.tests import DEVNULL
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_GETLOADAVG
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import IS_64BIT
from psutil.tests import MACOS_12PLUS
from psutil.tests import PYPY
from psutil.tests import UNICODE_SUFFIX
from psutil.tests import PsutilTestCase
from psutil.tests import check_net_address
from psutil.tests import enum
from psutil.tests import mock
from psutil.tests import retry_on_failure
def test_net_if_addrs(self):
    nics = psutil.net_if_addrs()
    assert nics, nics
    nic_stats = psutil.net_if_stats()
    families = set([socket.AF_INET, socket.AF_INET6, psutil.AF_LINK])
    for nic, addrs in nics.items():
        self.assertIsInstance(nic, str)
        self.assertEqual(len(set(addrs)), len(addrs))
        for addr in addrs:
            self.assertIsInstance(addr.family, int)
            self.assertIsInstance(addr.address, str)
            self.assertIsInstance(addr.netmask, (str, type(None)))
            self.assertIsInstance(addr.broadcast, (str, type(None)))
            self.assertIn(addr.family, families)
            if PY3 and (not PYPY):
                self.assertIsInstance(addr.family, enum.IntEnum)
            if nic_stats[nic].isup:
                if addr.family == socket.AF_INET:
                    s = socket.socket(addr.family)
                    with contextlib.closing(s):
                        s.bind((addr.address, 0))
                elif addr.family == socket.AF_INET6:
                    info = socket.getaddrinfo(addr.address, 0, socket.AF_INET6, socket.SOCK_STREAM, 0, socket.AI_PASSIVE)[0]
                    af, socktype, proto, canonname, sa = info
                    s = socket.socket(af, socktype, proto)
                    with contextlib.closing(s):
                        s.bind(sa)
            for ip in (addr.address, addr.netmask, addr.broadcast, addr.ptp):
                if ip is not None:
                    if addr.family != socket.AF_INET6:
                        check_net_address(ip, addr.family)
            if addr.broadcast:
                self.assertIsNone(addr.ptp)
            elif addr.ptp:
                self.assertIsNone(addr.broadcast)
    if BSD or MACOS or SUNOS:
        if hasattr(socket, 'AF_LINK'):
            self.assertEqual(psutil.AF_LINK, socket.AF_LINK)
    elif LINUX:
        self.assertEqual(psutil.AF_LINK, socket.AF_PACKET)
    elif WINDOWS:
        self.assertEqual(psutil.AF_LINK, -1)