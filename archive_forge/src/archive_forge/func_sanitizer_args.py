from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
def sanitizer_args(self, value: str) -> T.List[str]:
    if value == 'none':
        return []
    return ['-fsanitize=' + value]