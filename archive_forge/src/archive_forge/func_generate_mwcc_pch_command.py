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
def generate_mwcc_pch_command(self, target, compiler, pch):
    commands = self._generate_single_compile(target, compiler)
    dst = os.path.join(self.get_target_private_dir(target), os.path.basename(pch) + '.' + compiler.get_pch_suffix())
    dep = os.path.splitext(dst)[0] + '.' + compiler.get_depfile_suffix()
    return (commands, dep, dst, [])