from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from pathlib import PurePath, Path
from textwrap import dedent
import itertools
import json
import os
import pickle
import re
import subprocess
import typing as T
from . import backends
from .. import modules
from .. import environment, mesonlib
from .. import build
from .. import mlog
from .. import compilers
from ..arglist import CompilerArgs
from ..compilers import Compiler
from ..linkers import ArLikeLinker, RSPFileSyntax
from ..mesonlib import (
from ..mesonlib import get_compiler_for_source, has_path_sep, OptionKey
from .backends import CleanTrees
from ..build import GeneratedList, InvalidArguments
def get_dependency_filename(self, t):
    if isinstance(t, build.SharedLibrary):
        return self.get_target_shsym_filename(t)
    elif isinstance(t, mesonlib.File):
        if t.is_built:
            return t.relative_name()
        else:
            return t.absolute_path(self.environment.get_source_dir(), self.environment.get_build_dir())
    return self.get_target_filename(t)