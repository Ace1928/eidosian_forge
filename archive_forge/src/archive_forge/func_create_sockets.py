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
@contextlib.contextmanager
def create_sockets():
    """Open as many socket families / types as possible."""
    socks = []
    fname1 = fname2 = None
    try:
        socks.append(bind_socket(socket.AF_INET, socket.SOCK_STREAM))
        socks.append(bind_socket(socket.AF_INET, socket.SOCK_DGRAM))
        if supports_ipv6():
            socks.append(bind_socket(socket.AF_INET6, socket.SOCK_STREAM))
            socks.append(bind_socket(socket.AF_INET6, socket.SOCK_DGRAM))
        if POSIX and HAS_CONNECTIONS_UNIX:
            fname1 = get_testfn()
            fname2 = get_testfn()
            s1, s2 = unix_socketpair(fname1)
            s3 = bind_unix_socket(fname2, type=socket.SOCK_DGRAM)
            for s in (s1, s2, s3):
                socks.append(s)
        yield socks
    finally:
        for s in socks:
            s.close()
        for fname in (fname1, fname2):
            if fname is not None:
                safe_rmpath(fname)