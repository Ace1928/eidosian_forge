from __future__ import annotations
import math
import numbers
from collections.abc import Iterable
from enum import Enum
from typing import Any
from dask import config, core, utils
from dask.base import normalize_token, tokenize
from dask.core import (
from dask.typing import Graph, Key
def default_fused_linear_keys_renamer(keys):
    """Create new keys for fused tasks"""
    typ = type(keys[0])
    if typ is str:
        names = [utils.key_split(x) for x in keys[:0:-1]]
        names.append(keys[0])
        return '-'.join(names)
    elif typ is tuple and len(keys[0]) > 0 and isinstance(keys[0][0], str):
        names = [utils.key_split(x) for x in keys[:0:-1]]
        names.append(keys[0][0])
        return ('-'.join(names),) + keys[0][1:]
    else:
        return None