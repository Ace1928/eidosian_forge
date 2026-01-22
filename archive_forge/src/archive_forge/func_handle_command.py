from __future__ import annotations
from .common import CMakeException
from .generator import parse_generator_expressions
from .. import mlog
from ..mesonlib import version_compare
import typing as T
from pathlib import Path
from functools import lru_cache
import re
import json
import textwrap
def handle_command(key: str, target: CMakeGeneratorTarget) -> None:
    if key == 'ARGS':
        return
    target.command[-1] += [key]