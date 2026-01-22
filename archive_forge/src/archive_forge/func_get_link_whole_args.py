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
def get_link_whole_args(self, linker, target):
    use_custom = False
    if linker.id == 'msvc':
        if mesonlib.version_compare(linker.version, '<19.00.23918'):
            use_custom = True
    if use_custom:
        objects_from_static_libs: T.List[ExtractedObjects] = []
        for dep in target.link_whole_targets:
            l = dep.extract_all_objects(False)
            objects_from_static_libs += self.determine_ext_objs(l, '')
            objects_from_static_libs.extend(self.flatten_object_list(dep)[0])
        return objects_from_static_libs
    else:
        target_args = self.build_target_link_arguments(linker, target.link_whole_targets)
        return linker.get_link_whole_for(target_args) if target_args else []