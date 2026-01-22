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
def do_target(t: str) -> None:
    if t not in self.targets:
        return self._gen_exception('set_property', f'TARGET {t} not found', tline)
    tgt = self.targets[t]
    if identifier not in tgt.properties:
        tgt.properties[identifier] = []
    if append:
        tgt.properties[identifier] += value
    else:
        tgt.properties[identifier] = value