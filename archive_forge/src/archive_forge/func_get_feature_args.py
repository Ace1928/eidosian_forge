from __future__ import annotations
import abc
import contextlib, os.path, re
import enum
import itertools
import typing as T
from functools import lru_cache
from .. import coredata
from .. import mlog
from .. import mesonlib
from ..mesonlib import (
from ..arglist import CompilerArgs
def get_feature_args(self, kwargs: DFeatures, build_to_src: str) -> T.List[str]:
    """Used by D for extra language features."""
    raise EnvironmentException(f'{self.id} does not implement get_feature_args')