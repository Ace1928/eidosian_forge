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
def get_generated_headers(self, target):
    if hasattr(target, 'cached_generated_headers'):
        return target.cached_generated_headers
    header_deps = []
    for genlist in target.get_generated_sources():
        if isinstance(genlist, (build.CustomTarget, build.CustomTargetIndex)):
            continue
        for src in genlist.get_outputs():
            if self.environment.is_header(src):
                header_deps.append(self.get_target_generated_dir(target, genlist, src))
    if 'vala' in target.compilers and (not isinstance(target, build.Executable)):
        vala_header = File.from_built_file(self.get_target_dir(target), target.vala_header)
        header_deps.append(vala_header)
    for dep in itertools.chain(target.link_targets, target.link_whole_targets):
        if isinstance(dep, (build.StaticLibrary, build.SharedLibrary)):
            header_deps += self.get_generated_headers(dep)
    if isinstance(target, build.CompileTarget):
        header_deps.extend(target.get_generated_headers())
    target.cached_generated_headers = header_deps
    return header_deps