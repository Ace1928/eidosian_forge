from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
def compiler_to_generator(self, target: build.BuildTarget, compiler: 'Compiler', sources: _ALL_SOURCES_TYPE, output_templ: str, depends: T.Optional[T.List[T.Union[build.BuildTarget, build.CustomTarget, build.CustomTargetIndex]]]=None) -> build.GeneratedList:
    """
        Some backends don't support custom compilers. This is a convenience
        method to convert a Compiler to a Generator.
        """
    exelist = compiler.get_exelist()
    exe = programs.ExternalProgram(exelist[0])
    args = exelist[1:]
    commands = self.generate_basic_compiler_args(target, compiler)
    commands += compiler.get_dependency_gen_args('@OUTPUT@', '@DEPFILE@')
    commands += compiler.get_output_args('@OUTPUT@')
    commands += compiler.get_compile_only_args() + ['@INPUT@']
    commands += self.get_source_dir_include_args(target, compiler)
    commands += self.get_build_dir_include_args(target, compiler)
    commands += self.escape_extra_args(target.get_extra_args(compiler.get_language()))
    generator = build.Generator(exe, args + commands.to_native(), [output_templ], depfile='@PLAINNAME@.d', depends=depends)
    return generator.process_files(sources, self.interpreter)