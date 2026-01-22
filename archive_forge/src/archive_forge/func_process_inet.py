from __future__ import division
import base64
import collections
import errno
import functools
import glob
import os
import re
import socket
import struct
import sys
import warnings
from collections import defaultdict
from collections import namedtuple
from . import _common
from . import _psposix
from . import _psutil_linux as cext
from . import _psutil_posix as cext_posix
from ._common import NIC_DUPLEX_FULL
from ._common import NIC_DUPLEX_HALF
from ._common import NIC_DUPLEX_UNKNOWN
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import bcat
from ._common import cat
from ._common import debug
from ._common import decode
from ._common import get_procfs_path
from ._common import isfile_strict
from ._common import memoize
from ._common import memoize_when_activated
from ._common import open_binary
from ._common import open_text
from ._common import parse_environ_block
from ._common import path_exists_strict
from ._common import supports_ipv6
from ._common import usage_percent
from ._compat import PY3
from ._compat import FileNotFoundError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import b
from ._compat import basestring
@staticmethod
def process_inet(file, family, type_, inodes, filter_pid=None):
    """Parse /proc/net/tcp* and /proc/net/udp* files."""
    if file.endswith('6') and (not os.path.exists(file)):
        return
    with open_text(file) as f:
        f.readline()
        for lineno, line in enumerate(f, 1):
            try:
                _, laddr, raddr, status, _, _, _, _, _, inode = line.split()[:10]
            except ValueError:
                raise RuntimeError('error while parsing %s; malformed line %s %r' % (file, lineno, line))
            if inode in inodes:
                pid, fd = inodes[inode][0]
            else:
                pid, fd = (None, -1)
            if filter_pid is not None and filter_pid != pid:
                continue
            else:
                if type_ == socket.SOCK_STREAM:
                    status = TCP_STATUSES[status]
                else:
                    status = _common.CONN_NONE
                try:
                    laddr = Connections.decode_address(laddr, family)
                    raddr = Connections.decode_address(raddr, family)
                except _Ipv6UnsupportedError:
                    continue
                yield (fd, family, type_, laddr, raddr, status, pid)