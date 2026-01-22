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
def env_dict(self) -> dict[str, str]:
    """Return a dictionary of the environment variables used to create the container."""
    return dict(((item[0], item[1]) for item in [e.split('=', 1) for e in self.env]))