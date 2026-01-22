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
def generate_basic_compiler_args(self, target: build.BuildTarget, compiler: 'Compiler') -> 'CompilerArgs':
    commands = compiler.compiler_args()
    copt_proxy = target.get_options()
    commands += self.get_no_stdlib_args(target, compiler)
    commands += compiler.get_always_args()
    commands += compiler.get_warn_args(T.cast('str', target.get_option(OptionKey('warning_level'))))
    if target.get_option(OptionKey('werror')):
        commands += compiler.get_werror_args()
    commands += compiler.get_option_compile_args(copt_proxy)
    optimization = target.get_option(OptionKey('optimization'))
    assert isinstance(optimization, str), 'for mypy'
    commands += compiler.get_optimization_args(optimization)
    debug = target.get_option(OptionKey('debug'))
    assert isinstance(debug, bool), 'for mypy'
    commands += compiler.get_debug_args(debug)
    commands += self.build.get_project_args(compiler, target.subproject, target.for_machine)
    commands += self.build.get_global_args(compiler, target.for_machine)
    commands += self.environment.coredata.get_external_args(target.for_machine, compiler.get_language())
    if '/Zi' in commands and ('/ZI' in commands or '/Z7' in commands):
        commands.remove('/Zi')
    if isinstance(target, build.SharedLibrary):
        commands += compiler.get_pic_args()
    if isinstance(target, build.StaticLibrary) and target.pic:
        commands += compiler.get_pic_args()
    elif isinstance(target, (build.StaticLibrary, build.Executable)) and target.pie:
        commands += compiler.get_pie_args()
    for dep in reversed(target.get_external_deps()):
        if not dep.found():
            continue
        if compiler.language == 'vala':
            if dep.type_name == 'pkgconfig':
                assert isinstance(dep, dependencies.ExternalDependency)
                if dep.name == 'glib-2.0' and dep.version_reqs is not None:
                    for req in dep.version_reqs:
                        if req.startswith(('>=', '==')):
                            commands += ['--target-glib', req[2:]]
                            break
                commands += ['--pkg', dep.name]
            elif isinstance(dep, dependencies.ExternalLibrary):
                commands += dep.get_link_args('vala')
        else:
            commands += compiler.get_dependency_compile_args(dep)
        if isinstance(target, build.Executable):
            commands += dep.get_exe_args(compiler)
    if compiler.language == 'fortran':
        for lt in chain(target.link_targets, target.link_whole_targets):
            priv_dir = self.get_target_private_dir(lt)
            commands += compiler.get_include_args(priv_dir, False)
    return commands