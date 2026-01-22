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
def get_target_type_link_args(self, target, linker):
    commands = []
    if isinstance(target, build.Executable):
        commands += linker.get_std_exe_link_args()
        if target.export_dynamic:
            commands += linker.gen_export_dynamic_link_args(self.environment)
        if target.import_filename:
            commands += linker.gen_import_library_args(self.get_import_filename(target))
        if target.pie:
            commands += linker.get_pie_link_args()
        if target.vs_module_defs and hasattr(linker, 'gen_vs_module_defs_args'):
            commands += linker.gen_vs_module_defs_args(target.vs_module_defs.rel_to_builddir(self.build_to_src))
    elif isinstance(target, build.SharedLibrary):
        if isinstance(target, build.SharedModule):
            commands += linker.get_std_shared_module_link_args(target.get_options())
        else:
            commands += linker.get_std_shared_lib_link_args()
        commands += linker.get_pic_args()
        if not isinstance(target, build.SharedModule) or target.force_soname:
            commands += linker.get_soname_args(self.environment, target.prefix, target.name, target.suffix, target.soversion, target.darwin_versions)
        if target.vs_module_defs and hasattr(linker, 'gen_vs_module_defs_args'):
            commands += linker.gen_vs_module_defs_args(target.vs_module_defs.rel_to_builddir(self.build_to_src))
        if target.import_filename:
            commands += linker.gen_import_library_args(self.get_import_filename(target))
    elif isinstance(target, build.StaticLibrary):
        commands += linker.get_std_link_args(self.environment, not target.should_install())
    else:
        raise RuntimeError('Unknown build target type.')
    return commands