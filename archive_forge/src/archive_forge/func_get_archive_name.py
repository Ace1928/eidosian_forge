from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
def get_archive_name(self, filename: str) -> str:
    filename = re.sub('[.][a]([.]?([0-9]+))*([.]?([a-z]+))*', '.a', filename.replace('.so', '.a'))
    return filename