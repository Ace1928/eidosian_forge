from __future__ import annotations
import inspect
from contextlib import contextmanager
from typing import Any, Dict, Iterator
import param
from packaging.version import Version
def should_inherit(parameterized: param.Parameterized, p: str, v: Any) -> Any:
    pobj = parameterized.param[p]
    return v is not pobj.default and (not pobj.readonly) and (v is not None or pobj.allow_None)