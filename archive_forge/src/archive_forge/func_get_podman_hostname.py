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
@cache
def get_podman_hostname() -> str:
    """Return the hostname of the Podman service."""
    hostname = get_podman_remote()
    if not hostname:
        hostname = 'localhost'
        display.info('Assuming Podman is available on localhost.', verbosity=1)
    return hostname