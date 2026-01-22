from __future__ import annotations
import getpass
import hashlib
import json
import os
import pkgutil
import re
import sys
import time
import typing as t
import uuid
from contextlib import ExitStack
from io import BytesIO
from itertools import chain
from os.path import basename
from os.path import join
from zlib import adler32
from .._internal import _log
from ..exceptions import NotFound
from ..http import parse_cookie
from ..security import gen_salt
from ..utils import send_file
from ..wrappers.request import Request
from ..wrappers.response import Response
from .console import Console
from .tbtools import DebugFrameSummary
from .tbtools import DebugTraceback
from .tbtools import render_console_html
def get_machine_id() -> str | bytes | None:
    global _machine_id
    if _machine_id is not None:
        return _machine_id

    def _generate() -> str | bytes | None:
        linux = b''
        for filename in ('/etc/machine-id', '/proc/sys/kernel/random/boot_id'):
            try:
                with open(filename, 'rb') as f:
                    value = f.readline().strip()
            except OSError:
                continue
            if value:
                linux += value
                break
        try:
            with open('/proc/self/cgroup', 'rb') as f:
                linux += f.readline().strip().rpartition(b'/')[2]
        except OSError:
            pass
        if linux:
            return linux
        try:
            from subprocess import Popen, PIPE
            dump = Popen(['ioreg', '-c', 'IOPlatformExpertDevice', '-d', '2'], stdout=PIPE).communicate()[0]
            match = re.search(b'"serial-number" = <([^>]+)', dump)
            if match is not None:
                return match.group(1)
        except (OSError, ImportError):
            pass
        if sys.platform == 'win32':
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\Microsoft\\Cryptography', 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as rk:
                    guid: str | bytes
                    guid_type: int
                    guid, guid_type = winreg.QueryValueEx(rk, 'MachineGuid')
                    if guid_type == winreg.REG_SZ:
                        return guid.encode('utf-8')
                    return guid
            except OSError:
                pass
        return None
    _machine_id = _generate()
    return _machine_id