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
def copyload_shared_lib(suffix=''):
    """Ctx manager which picks up a random shared DLL lib used
        by this process, copies it in another location and loads it
        in memory via ctypes.
        Return the new absolutized, normcased path.
        """
    from ctypes import WinError
    from ctypes import wintypes
    ext = '.dll'
    dst = get_testfn(suffix=suffix + ext)
    libs = [x.path for x in psutil.Process().memory_maps() if x.path.lower().endswith(ext) and 'python' in os.path.basename(x.path).lower() and ('wow64' not in x.path.lower())]
    if PYPY and (not libs):
        libs = [x.path for x in psutil.Process().memory_maps() if 'pypy' in os.path.basename(x.path).lower()]
    src = random.choice(libs)
    shutil.copyfile(src, dst)
    cfile = None
    try:
        cfile = ctypes.WinDLL(dst)
        yield dst
    finally:
        if cfile is not None:
            FreeLibrary = ctypes.windll.kernel32.FreeLibrary
            FreeLibrary.argtypes = [wintypes.HMODULE]
            ret = FreeLibrary(cfile._handle)
            if ret == 0:
                WinError()
        safe_rmpath(dst)