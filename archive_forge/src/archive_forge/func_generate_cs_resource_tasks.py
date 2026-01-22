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
def generate_cs_resource_tasks(self, target):
    args = []
    deps = []
    for r in target.resources:
        rel_sourcefile = os.path.join(self.build_to_src, target.subdir, r)
        if r.endswith('.resources'):
            a = '-resource:' + rel_sourcefile
        elif r.endswith('.txt') or r.endswith('.resx'):
            ofilebase = os.path.splitext(os.path.basename(r))[0] + '.resources'
            ofilename = os.path.join(self.get_target_private_dir(target), ofilebase)
            elem = NinjaBuildElement(self.all_outputs, ofilename, 'CUSTOM_COMMAND', rel_sourcefile)
            elem.add_item('COMMAND', ['resgen', rel_sourcefile, ofilename])
            elem.add_item('DESC', f'Compiling resource {rel_sourcefile}')
            self.add_build(elem)
            deps.append(ofilename)
            a = '-resource:' + ofilename
        else:
            raise InvalidArguments(f'Unknown resource file {r}.')
        args.append(a)
    return (args, deps)