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
def get_tcp_port(self, port: int) -> t.Optional[list[dict[str, str]]]:
    """Return a list of the endpoints published by the container for the specified TCP port, or None if it is not published."""
    return self.ports.get('%d/tcp' % port)