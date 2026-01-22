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
def determine_dep_vapis(self, target):
    """
        Peek into the sources of BuildTargets we're linking with, and if any of
        them was built with Vala, assume that it also generated a .vapi file of
        the same name as the BuildTarget and return the path to it relative to
        the build directory.
        """
    result = OrderedSet()
    for dep in itertools.chain(target.link_targets, target.link_whole_targets):
        if not dep.is_linkable_target():
            continue
        for i in dep.sources:
            if hasattr(i, 'fname'):
                i = i.fname
            if i.split('.')[-1] in compilers.lang_suffixes['vala']:
                vapiname = dep.vala_vapi
                fullname = os.path.join(self.get_target_dir(dep), vapiname)
                result.add(fullname)
                break
    return list(result)