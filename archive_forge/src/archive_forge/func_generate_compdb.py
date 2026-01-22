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
def generate_compdb(self):
    rules = []
    for for_machine in MachineChoice:
        for compiler in self.environment.coredata.compilers[for_machine].values():
            rules += [f'{rule}{ext}' for rule in [self.compiler_to_rule_name(compiler)] for ext in ['', '_RSP']]
            rules += [f'{rule}{ext}' for rule in [self.compiler_to_pch_rule_name(compiler)] for ext in ['', '_RSP']]
    compdb_options = ['-x'] if mesonlib.version_compare(self.ninja_version, '>=1.9') else []
    ninja_compdb = self.ninja_command + ['-t', 'compdb'] + compdb_options + rules
    builddir = self.environment.get_build_dir()
    try:
        jsondb = subprocess.check_output(ninja_compdb, cwd=builddir)
        with open(os.path.join(builddir, 'compile_commands.json'), 'wb') as f:
            f.write(jsondb)
    except Exception:
        mlog.warning('Could not create compilation database.', fatal=False)