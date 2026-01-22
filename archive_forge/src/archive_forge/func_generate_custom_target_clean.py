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
def generate_custom_target_clean(self, trees: T.List[str]) -> str:
    e = self.create_phony_target('clean-ctlist', 'CUSTOM_COMMAND', 'PHONY')
    d = CleanTrees(self.environment.get_build_dir(), trees)
    d_file = os.path.join(self.environment.get_scratch_dir(), 'cleantrees.dat')
    e.add_item('COMMAND', self.environment.get_build_command() + ['--internal', 'cleantrees', d_file])
    e.add_item('description', 'Cleaning custom target directories')
    self.add_build(e)
    with open(d_file, 'wb') as ofile:
        pickle.dump(d, ofile)
    return 'clean-ctlist'