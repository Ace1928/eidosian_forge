from __future__ import annotations
import dataclasses
import enum
import functools
import inspect
from inspect import Parameter
from inspect import signature
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Final
from typing import NoReturn
import py
def getlocation(function, curdir: str | os.PathLike[str] | None=None) -> str:
    function = get_real_func(function)
    fn = Path(inspect.getfile(function))
    lineno = function.__code__.co_firstlineno
    if curdir is not None:
        try:
            relfn = fn.relative_to(curdir)
        except ValueError:
            pass
        else:
            return '%s:%d' % (relfn, lineno + 1)
    return '%s:%d' % (fn, lineno + 1)