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
def var_to_str(self, var: str) -> T.Optional[str]:
    if var in self.vars and self.vars[var]:
        return self.vars[var][0]
    return None