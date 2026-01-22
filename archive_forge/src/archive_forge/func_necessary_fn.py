from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def necessary_fn(x: Any) -> Any:
    return x[x[field] != ''].dropna(subset=[field])