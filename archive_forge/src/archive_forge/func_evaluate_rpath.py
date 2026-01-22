from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
def evaluate_rpath(p: str, build_dir: str, from_dir: str) -> str:
    if p == from_dir:
        return ''
    elif os.path.isabs(p):
        return p
    else:
        return os.path.relpath(os.path.join(build_dir, p), os.path.join(build_dir, from_dir))