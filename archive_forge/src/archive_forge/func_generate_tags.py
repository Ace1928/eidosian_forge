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
def generate_tags(self, tool: str, target_name: str) -> None:
    import shutil
    if not shutil.which(tool):
        return
    if target_name in self.all_outputs:
        return
    cmd = self.environment.get_build_command() + ['--internal', 'tags', tool, self.environment.source_dir]
    elem = self.create_phony_target(target_name, 'CUSTOM_COMMAND', 'PHONY')
    elem.add_item('COMMAND', cmd)
    elem.add_item('pool', 'console')
    self.add_build(elem)