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
def docker_desktop_wsl2(self) -> bool:
    """Return True if Docker Desktop integrated with WSL2 is detected, otherwise False."""
    info = self.info
    kernel_version = info.get('KernelVersion')
    operating_system = info.get('OperatingSystem')
    dd_wsl2 = kernel_version and kernel_version.endswith('-WSL2') and (operating_system == 'Docker Desktop')
    return dd_wsl2