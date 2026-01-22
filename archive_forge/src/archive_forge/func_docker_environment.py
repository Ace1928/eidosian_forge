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
def docker_environment() -> dict[str, str]:
    """Return a dictionary of docker related environment variables found in the current environment."""
    env = common_environment()
    var_names = {'XDG_RUNTIME_DIR'}
    var_prefixes = {'CONTAINER_', 'DOCKER_'}
    env.update({name: value for name, value in os.environ.items() if name in var_names or any((name.startswith(prefix) for prefix in var_prefixes))})
    return env