from __future__ import print_function
import atexit
import contextlib
import ctypes
import errno
import functools
import gc
import inspect
import os
import platform
import random
import re
import select
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import warnings
from socket import AF_INET
from socket import AF_INET6
from socket import SOCK_STREAM
import psutil
from psutil import AIX
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import bytes2human
from psutil._common import debug
from psutil._common import memoize
from psutil._common import print_color
from psutil._common import supports_ipv6
from psutil._compat import PY3
from psutil._compat import FileExistsError
from psutil._compat import FileNotFoundError
from psutil._compat import range
from psutil._compat import super
from psutil._compat import u
from psutil._compat import unicode
from psutil._compat import which
def check_connection_ntuple(conn):
    """Check validity of a connection namedtuple."""

    def check_ntuple(conn):
        has_pid = len(conn) == 7
        assert len(conn) in (6, 7), len(conn)
        assert conn[0] == conn.fd, conn.fd
        assert conn[1] == conn.family, conn.family
        assert conn[2] == conn.type, conn.type
        assert conn[3] == conn.laddr, conn.laddr
        assert conn[4] == conn.raddr, conn.raddr
        assert conn[5] == conn.status, conn.status
        if has_pid:
            assert conn[6] == conn.pid, conn.pid

    def check_family(conn):
        assert conn.family in (AF_INET, AF_INET6, AF_UNIX), conn.family
        if enum is not None:
            assert isinstance(conn.family, enum.IntEnum), conn
        else:
            assert isinstance(conn.family, int), conn
        if conn.family == AF_INET:
            s = socket.socket(conn.family, conn.type)
            with contextlib.closing(s):
                try:
                    s.bind((conn.laddr[0], 0))
                except socket.error as err:
                    if err.errno != errno.EADDRNOTAVAIL:
                        raise
        elif conn.family == AF_UNIX:
            assert conn.status == psutil.CONN_NONE, conn.status

    def check_type(conn):
        SOCK_SEQPACKET = getattr(socket, 'SOCK_SEQPACKET', object())
        assert conn.type in (socket.SOCK_STREAM, socket.SOCK_DGRAM, SOCK_SEQPACKET), conn.type
        if enum is not None:
            assert isinstance(conn.type, enum.IntEnum), conn
        else:
            assert isinstance(conn.type, int), conn
        if conn.type == socket.SOCK_DGRAM:
            assert conn.status == psutil.CONN_NONE, conn.status

    def check_addrs(conn):
        for addr in (conn.laddr, conn.raddr):
            if conn.family in (AF_INET, AF_INET6):
                assert isinstance(addr, tuple), type(addr)
                if not addr:
                    continue
                assert isinstance(addr.port, int), type(addr.port)
                assert 0 <= addr.port <= 65535, addr.port
                check_net_address(addr.ip, conn.family)
            elif conn.family == AF_UNIX:
                assert isinstance(addr, str), type(addr)

    def check_status(conn):
        assert isinstance(conn.status, str), conn.status
        valids = [getattr(psutil, x) for x in dir(psutil) if x.startswith('CONN_')]
        assert conn.status in valids, conn.status
        if conn.family in (AF_INET, AF_INET6) and conn.type == SOCK_STREAM:
            assert conn.status != psutil.CONN_NONE, conn.status
        else:
            assert conn.status == psutil.CONN_NONE, conn.status
    check_ntuple(conn)
    check_family(conn)
    check_type(conn)
    check_addrs(conn)
    check_status(conn)