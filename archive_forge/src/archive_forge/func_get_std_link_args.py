from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
def get_std_link_args(self, env: 'Environment', is_thin: bool) -> T.List[str]:
    return self.std_args