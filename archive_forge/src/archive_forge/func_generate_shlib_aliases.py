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
def generate_shlib_aliases(self, target, outdir):
    for alias, to, tag in target.get_aliases():
        aliasfile = os.path.join(outdir, alias)
        abs_aliasfile = os.path.join(self.environment.get_build_dir(), outdir, alias)
        try:
            os.remove(abs_aliasfile)
        except Exception:
            pass
        try:
            os.symlink(to, abs_aliasfile)
        except NotImplementedError:
            mlog.debug('Library versioning disabled because symlinks are not supported.')
        except OSError:
            mlog.debug('Library versioning disabled because we do not have symlink creation privileges.')
        else:
            self.implicit_meson_outs.append(aliasfile)