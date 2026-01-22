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
def network_settings(self) -> dict[str, t.Any]:
    """Return a dictionary of the container network settings."""
    return self.inspection['NetworkSettings']