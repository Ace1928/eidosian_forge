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
def docker_rm(args: CommonConfig, container_id: str) -> None:
    """Remove the specified container."""
    try:
        docker_command(args, ['stop', '--time', '0', container_id], capture=True)
        docker_command(args, ['rm', container_id], capture=True)
    except SubprocessError as ex:
        if 'no such container' not in ex.stderr.lower():
            raise ex