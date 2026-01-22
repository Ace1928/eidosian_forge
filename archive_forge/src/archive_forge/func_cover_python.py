from __future__ import annotations
import dataclasses
import os
import sqlite3
import tempfile
import typing as t
from .config import (
from .io import (
from .util import (
from .data import (
from .util_common import (
from .host_configs import (
from .constants import (
from .thread import (
def cover_python(args: TestConfig, python: PythonConfig, cmd: list[str], target_name: str, env: dict[str, str], capture: bool, data: t.Optional[str]=None, cwd: t.Optional[str]=None) -> tuple[t.Optional[str], t.Optional[str]]:
    """Run a command while collecting Python code coverage."""
    if args.coverage:
        env.update(get_coverage_environment(args, target_name, python.version))
    return intercept_python(args, python, cmd, env, capture, data, cwd)