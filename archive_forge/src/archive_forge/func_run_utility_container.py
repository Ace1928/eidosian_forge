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
def run_utility_container(args: CommonConfig, name: str, cmd: list[str], options: list[str], data: t.Optional[str]=None) -> tuple[t.Optional[str], t.Optional[str]]:
    """Run the specified command using the ansible-test utility container, returning stdout and stderr."""
    name = get_session_container_name(args, name)
    options = options + ['--name', name, '--rm']
    if data:
        options.append('-i')
    docker_pull(args, UTILITY_IMAGE)
    return docker_run(args, UTILITY_IMAGE, options, cmd, data)