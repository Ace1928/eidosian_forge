from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
def get_command_to_archive_shlib(self) -> T.List[str]:
    command = ['ar', '-q', '-v', '$out', '$in', '&&', 'rm', '-f', '$in']
    return command