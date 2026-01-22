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
def strip_properties(self) -> None:
    if not self.properties:
        return
    for key, val in self.properties.items():
        self.properties[key] = [x.strip() for x in val]
        assert all((';' not in x for x in self.properties[key]))