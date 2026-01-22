from __future__ import annotations
from contextlib import redirect_stdout
import collections
import dataclasses
import json
import os
from pathlib import Path, PurePath
import sys
import typing as T
from . import build, mesonlib, coredata as cdata
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstJSONPrinter
from .backend import backends
from .dependencies import Dependency
from . import environment
from .interpreterbase import ObjectHolder
from .mesonlib import OptionKey
from .mparser import FunctionNode, ArrayNode, ArgumentNode, BaseStringNode
def list_deps(coredata: cdata.CoreData, backend: backends.Backend) -> T.List[T.Dict[str, T.Union[str, T.List[str]]]]:
    result: T.Dict[str, T.Dict[str, T.Union[str, T.List[str]]]] = {}

    def _src_to_str(src_file: T.Union[mesonlib.FileOrString, build.CustomTarget, build.StructuredSources, build.CustomTargetIndex, build.GeneratedList]) -> T.List[str]:
        if isinstance(src_file, str):
            return [src_file]
        if isinstance(src_file, mesonlib.File):
            return [src_file.absolute_path(backend.source_dir, backend.build_dir)]
        if isinstance(src_file, (build.CustomTarget, build.CustomTargetIndex, build.GeneratedList)):
            return src_file.get_outputs()
        if isinstance(src_file, build.StructuredSources):
            return [f for s in src_file.as_list() for f in _src_to_str(s)]
        raise mesonlib.MesonBugException(f'Invalid file type {type(src_file)}.')

    def _create_result(d: Dependency, varname: T.Optional[str]=None) -> T.Dict[str, T.Any]:
        return {'name': d.name, 'type': d.type_name, 'version': d.get_version(), 'compile_args': d.get_compile_args(), 'link_args': d.get_link_args(), 'include_directories': [i for idirs in d.get_include_dirs() for i in idirs.to_string_list(backend.source_dir, backend.build_dir)], 'sources': [f for s in d.get_sources() for f in _src_to_str(s)], 'extra_files': [f for s in d.get_extra_files() for f in _src_to_str(s)], 'dependencies': [e.name for e in d.ext_deps], 'depends': [lib.get_id() for lib in getattr(d, 'libraries', [])], 'meson_variables': [varname] if varname else []}
    for d in coredata.deps.host.values():
        if d.found():
            result[d.name] = _create_result(d)
    for varname, holder in backend.interpreter.variables.items():
        if isinstance(holder, ObjectHolder):
            d = holder.held_object
            if isinstance(d, Dependency) and d.found():
                if d.name in result:
                    T.cast('T.List[str]', result[d.name]['meson_variables']).append(varname)
                else:
                    result[d.name] = _create_result(d, varname)
    return list(result.values())