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
def generate_single_compile(self, target: build.BuildTarget, src, is_generated: bool=False, header_deps=None, order_deps: T.Optional[T.List['mesonlib.FileOrString']]=None, extra_args: T.Optional[T.List[str]]=None, unity_sources: T.Optional[T.List[mesonlib.FileOrString]]=None) -> None:
    """
        Compiles C/C++, ObjC/ObjC++, Fortran, and D sources
        """
    header_deps = header_deps if header_deps is not None else []
    order_deps = order_deps if order_deps is not None else []
    if isinstance(src, str) and src.endswith('.h'):
        raise AssertionError(f'BUG: sources should not contain headers {src!r}')
    compiler = get_compiler_for_source(target.compilers.values(), src)
    commands = self._generate_single_compile_base_args(target, compiler)
    use_pch = self.target_uses_pch(target)
    if use_pch and 'mw' not in compiler.id:
        commands += self.get_pch_include_args(compiler, target)
    commands += self._generate_single_compile_target_args(target, compiler)
    if use_pch and 'mw' in compiler.id:
        commands += self.get_pch_include_args(compiler, target)
    commands = commands.compiler.compiler_args(commands)
    if is_generated is False:
        self.create_target_source_introspection(target, compiler, commands, [src], [], unity_sources)
    else:
        self.create_target_source_introspection(target, compiler, commands, [], [src], unity_sources)
    build_dir = self.environment.get_build_dir()
    if isinstance(src, File):
        rel_src = src.rel_to_builddir(self.build_to_src)
        if os.path.isabs(rel_src):
            if src.is_built:
                assert rel_src.startswith(build_dir)
                rel_src = rel_src[len(build_dir) + 1:]
    elif is_generated:
        raise AssertionError(f'BUG: broken generated source file handling for {src!r}')
    else:
        raise InvalidArguments(f'Invalid source type: {src!r}')
    obj_basename = self.object_filename_from_source(target, src)
    rel_obj = os.path.join(self.get_target_private_dir(target), obj_basename)
    dep_file = compiler.depfile_for_object(rel_obj)
    commands += self.get_compile_debugfile_args(compiler, target, rel_obj)
    if self.target_uses_pch(target):
        pchlist = target.get_pch(compiler.language)
    else:
        pchlist = []
    if not pchlist:
        pch_dep = []
    elif compiler.id == 'intel':
        pch_dep = []
    else:
        arr = []
        i = os.path.join(self.get_target_private_dir(target), compiler.get_pch_name(pchlist[0]))
        arr.append(i)
        pch_dep = arr
    compiler_name = self.compiler_to_rule_name(compiler)
    extra_deps = []
    if compiler.get_language() == 'fortran':
        if not is_generated:
            abs_src = Path(build_dir) / rel_src
            extra_deps += self.get_fortran_deps(compiler, abs_src, target)
        if not self.use_dyndeps_for_fortran():
            for modname, srcfile in self.fortran_deps[target.get_basename()].items():
                modfile = os.path.join(self.get_target_private_dir(target), compiler.module_name_to_filename(modname))
                if srcfile == src:
                    crstr = self.get_rule_suffix(target.for_machine)
                    depelem = NinjaBuildElement(self.all_outputs, modfile, 'FORTRAN_DEP_HACK' + crstr, rel_obj)
                    self.add_build(depelem)
        commands += compiler.get_module_outdir_args(self.get_target_private_dir(target))
    if extra_args is not None:
        commands.extend(extra_args)
    element = NinjaBuildElement(self.all_outputs, rel_obj, compiler_name, rel_src)
    self.add_header_deps(target, element, header_deps)
    for d in extra_deps:
        element.add_dep(d)
    for d in order_deps:
        if isinstance(d, File):
            d = d.rel_to_builddir(self.build_to_src)
        elif not self.has_dir_part(d):
            d = os.path.join(self.get_target_private_dir(target), d)
        element.add_orderdep(d)
    element.add_dep(pch_dep)
    for i in self.get_fortran_orderdeps(target, compiler):
        element.add_orderdep(i)
    if dep_file:
        element.add_item('DEPFILE', dep_file)
    if compiler.get_language() == 'cuda':

        def quote_make_target(targetName: str) -> str:
            result = ''
            for i, c in enumerate(targetName):
                if c in {' ', '\t'}:
                    for j in range(i - 1, -1, -1):
                        if targetName[j] == '\\':
                            result += '\\'
                        else:
                            break
                    result += '\\'
                elif c == '$':
                    result += '$'
                elif c == '#':
                    result += '\\'
                result += c
            return result
        element.add_item('CUDA_ESCAPED_TARGET', quote_make_target(rel_obj))
    element.add_item('ARGS', commands)
    self.add_dependency_scanner_entries_to_element(target, compiler, element, src)
    self.add_build(element)
    assert isinstance(rel_obj, str)
    assert isinstance(rel_src, str)
    return (rel_obj, rel_src.replace('\\', '/'))