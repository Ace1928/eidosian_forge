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
def decode_address(addr, family):
    """Accept an "ip:port" address as displayed in /proc/net/*
        and convert it into a human readable form, like:

        "0500000A:0016" -> ("10.0.0.5", 22)
        "0000000000000000FFFF00000100007F:9E49" -> ("::ffff:127.0.0.1", 40521)

        The IP address portion is a little or big endian four-byte
        hexadecimal number; that is, the least significant byte is listed
        first, so we need to reverse the order of the bytes to convert it
        to an IP address.
        The port is represented as a two-byte hexadecimal number.

        Reference:
        http://linuxdevcenter.com/pub/a/linux/2000/11/16/LinuxAdmin.html
        """
    ip, port = addr.split(':')
    port = int(port, 16)
    if not port:
        return ()
    if PY3:
        ip = ip.encode('ascii')
    if family == socket.AF_INET:
        if LITTLE_ENDIAN:
            ip = socket.inet_ntop(family, base64.b16decode(ip)[::-1])
        else:
            ip = socket.inet_ntop(family, base64.b16decode(ip))
    else:
        ip = base64.b16decode(ip)
        try:
            if LITTLE_ENDIAN:
                ip = socket.inet_ntop(socket.AF_INET6, struct.pack('>4I', *struct.unpack('<4I', ip)))
            else:
                ip = socket.inet_ntop(socket.AF_INET6, struct.pack('<4I', *struct.unpack('<4I', ip)))
        except ValueError:
            if not supports_ipv6():
                raise _Ipv6UnsupportedError
            else:
                raise
    return _common.addr(ip, port)