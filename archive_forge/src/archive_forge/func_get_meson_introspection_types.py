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
def get_meson_introspection_types(coredata: T.Optional[cdata.CoreData]=None, builddata: T.Optional[build.Build]=None, backend: T.Optional[backends.Backend]=None) -> 'T.Mapping[str, IntroCommand]':
    if backend and builddata:
        benchmarkdata = backend.create_test_serialisation(builddata.get_benchmarks())
        testdata = backend.create_test_serialisation(builddata.get_tests())
        installdata = backend.create_install_data()
        interpreter = backend.interpreter
    else:
        benchmarkdata = testdata = installdata = None
    return collections.OrderedDict([('ast', IntroCommand('Dump the AST of the meson file', no_bd=dump_ast)), ('benchmarks', IntroCommand('List all benchmarks', func=lambda: list_benchmarks(benchmarkdata))), ('buildoptions', IntroCommand('List all build options', func=lambda: list_buildoptions(coredata), no_bd=list_buildoptions_from_source)), ('buildsystem_files', IntroCommand('List files that make up the build system', func=lambda: list_buildsystem_files(builddata, interpreter))), ('compilers', IntroCommand('List used compilers', func=lambda: list_compilers(coredata))), ('dependencies', IntroCommand('List external dependencies', func=lambda: list_deps(coredata, backend), no_bd=list_deps_from_source)), ('scan_dependencies', IntroCommand('Scan for dependencies used in the meson.build file', no_bd=list_deps_from_source)), ('installed', IntroCommand('List all installed files and directories', func=lambda: list_installed(installdata))), ('install_plan', IntroCommand('List all installed files and directories with their details', func=lambda: list_install_plan(installdata))), ('machines', IntroCommand('Information about host, build, and target machines', func=lambda: list_machines(builddata))), ('projectinfo', IntroCommand('Information about projects', func=lambda: list_projinfo(builddata), no_bd=list_projinfo_from_source)), ('targets', IntroCommand('List top level targets', func=lambda: list_targets(builddata, installdata, backend), no_bd=list_targets_from_source)), ('tests', IntroCommand('List all unit tests', func=lambda: list_tests(testdata)))])