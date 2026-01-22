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
def generate_coverage_rules(self, gcovr_exe: T.Optional[str], gcovr_version: T.Optional[str], llvm_cov_exe: T.Optional[str]):
    e = self.create_phony_target('coverage', 'CUSTOM_COMMAND', 'PHONY')
    self.generate_coverage_command(e, [], gcovr_exe, llvm_cov_exe)
    e.add_item('description', 'Generates coverage reports')
    self.add_build(e)
    self.generate_coverage_legacy_rules(gcovr_exe, gcovr_version, llvm_cov_exe)