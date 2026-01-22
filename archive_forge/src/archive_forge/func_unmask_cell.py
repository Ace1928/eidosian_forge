import ast
import collections
import dataclasses
import secrets
import sys
from functools import lru_cache
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple
from black.output import out
from black.report import NothingChanged
def unmask_cell(src: str, replacements: List[Replacement]) -> str:
    """Remove replacements from cell.

    For example

        "9b20"
        foo = bar

    becomes

        %%time
        foo = bar
    """
    for replacement in replacements:
        src = src.replace(replacement.mask, replacement.src)
    return src