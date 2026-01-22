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
def generate_prelink(self, target, obj_list):
    assert isinstance(target, build.StaticLibrary)
    prelink_name = os.path.join(self.get_target_private_dir(target), target.name + '-prelink.o')
    elem = NinjaBuildElement(self.all_outputs, [prelink_name], 'CUSTOM_COMMAND', obj_list)
    prelinker = target.get_prelinker()
    cmd = prelinker.exelist[:]
    cmd += prelinker.get_prelink_args(prelink_name, obj_list)
    cmd = self.replace_paths(target, cmd)
    elem.add_item('COMMAND', cmd)
    elem.add_item('description', f'Prelinking {prelink_name}.')
    self.add_build(elem)
    return [prelink_name]