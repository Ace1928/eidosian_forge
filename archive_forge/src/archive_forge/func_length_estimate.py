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
def length_estimate(self, infiles, outfiles, elems):
    ninja_vars = {}
    for e in elems:
        name, value = e
        ninja_vars[name] = value
    ninja_vars['deps'] = self.deps
    ninja_vars['depfile'] = self.depfile
    ninja_vars['in'] = infiles
    ninja_vars['out'] = outfiles
    command = ' '.join([self._quoter(x) for x in self.command + self.args])
    estimate = len(command)
    for m in re.finditer('(\\${\\w+}|\\$\\w+)?[^$]*', command):
        if m.start(1) != -1:
            estimate -= m.end(1) - m.start(1) + 1
            chunk = m.group(1)
            if chunk[1] == '{':
                chunk = chunk[2:-1]
            else:
                chunk = chunk[1:]
            chunk = ninja_vars.get(chunk, [])
            estimate += len(' '.join(chunk))
    return estimate