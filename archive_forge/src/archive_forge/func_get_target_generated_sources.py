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
def get_target_generated_sources(self, target: build.BuildTarget) -> T.MutableMapping[str, File]:
    """
        Returns a dictionary with the keys being the path to the file
        (relative to the build directory) and the value being the File object
        representing the same path.
        """
    srcs: T.MutableMapping[str, File] = OrderedDict()
    for gensrc in target.get_generated_sources():
        for s in gensrc.get_outputs():
            rel_src = self.get_target_generated_dir(target, gensrc, s)
            srcs[rel_src] = File.from_built_relative(rel_src)
    return srcs