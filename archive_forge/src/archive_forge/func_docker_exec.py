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
def docker_exec(args: CommonConfig, container_id: str, cmd: list[str], capture: bool, options: t.Optional[list[str]]=None, stdin: t.Optional[t.IO[bytes]]=None, stdout: t.Optional[t.IO[bytes]]=None, interactive: bool=False, output_stream: t.Optional[OutputStream]=None, data: t.Optional[str]=None) -> tuple[t.Optional[str], t.Optional[str]]:
    """Execute the given command in the specified container."""
    if not options:
        options = []
    if data or stdin or stdout:
        options.append('-i')
    return docker_command(args, ['exec'] + options + [container_id] + cmd, capture=capture, stdin=stdin, stdout=stdout, interactive=interactive, output_stream=output_stream, data=data)