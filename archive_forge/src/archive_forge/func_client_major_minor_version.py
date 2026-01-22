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
def client_major_minor_version(self) -> tuple[int, int]:
    """The client major and minor version."""
    major, minor = self.client_version.split('.')[:2]
    return (int(major), int(minor))