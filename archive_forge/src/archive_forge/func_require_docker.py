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
def require_docker() -> DockerCommand:
    """Return the docker command to invoke. Raises an exception if docker is not available."""
    if (command := get_docker_command()):
        return command
    raise ApplicationError(f'No container runtime detected. Supported commands: {', '.join(DOCKER_COMMANDS)}')