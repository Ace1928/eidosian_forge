from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import re
import socket
import time
import urllib.parse
import typing as t
from .util import (
from .util_common import (
from .config import (
from .thread import (
from .cgroup import (
@property
def cgroup_version(self) -> int:
    """The cgroup version of the container host."""
    info = self.info
    host = info.get('host')
    if host:
        return int(host['cgroupVersion'].lstrip('v'))
    try:
        return int(info['CgroupVersion'])
    except KeyError:
        pass
    if self.server_major_minor_version < (20, 10):
        return 1
    message = f'The Docker client version is {self.client_version}. The Docker server version is {self.server_version}. Upgrade your Docker client to version 20.10 or later.'
    if detect_host_properties(self.args).cgroup_v2:
        raise ApplicationError(f'Unsupported Docker client and server combination using cgroup v2. {message}')
    display.warning(f'Detected Docker server cgroup v1 using probing. {message}', unique=True)
    return 1