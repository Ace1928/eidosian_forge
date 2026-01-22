from __future__ import annotations
import inspect
from contextlib import contextmanager
from typing import Any, Dict, Iterator
import param
from packaging.version import Version
def param_watchers(parameterized: param.Parameterized, value=_unset):
    if Version(param.__version__) <= Version('2.0.0a2'):
        if value is not _unset:
            parameterized._param_watchers = value
        else:
            return parameterized._param_watchers
    elif value is not _unset:
        parameterized.param.watchers = value
    else:
        return parameterized.param.watchers