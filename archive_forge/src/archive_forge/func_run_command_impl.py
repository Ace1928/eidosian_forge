from __future__ import annotations
from .. import mparser
from .. import environment
from .. import coredata
from .. import dependencies
from .. import mlog
from .. import build
from .. import optinterpreter
from .. import compilers
from .. import envconfig
from ..wrap import wrap, WrapMode
from .. import mesonlib
from ..mesonlib import (EnvironmentVariables, ExecutableSerialisation, MesonBugException, MesonException, HoldableObject,
from ..programs import ExternalProgram, NonExistingExternalProgram
from ..dependencies import Dependency
from ..depfile import DepFile
from ..interpreterbase import ContainerTypeInfo, InterpreterBase, KwargInfo, typed_kwargs, typed_pos_args
from ..interpreterbase import noPosargs, noKwargs, permittedKwargs, noArgsFlattening, noSecondLevelHolderResolving, unholder_return
from ..interpreterbase import InterpreterException, InvalidArguments, InvalidCode, SubdirDoneRequest
from ..interpreterbase import Disabler, disablerIfNotFound
from ..interpreterbase import FeatureNew, FeatureDeprecated, FeatureBroken, FeatureNewKwargs
from ..interpreterbase import ObjectHolder, ContextManagerObject
from ..interpreterbase import stringifyUserArguments
from ..modules import ExtensionModule, ModuleObject, MutableModuleObject, NewExtensionModule, NotFoundExtensionModule
from ..optinterpreter import optname_regex
from . import interpreterobjects as OBJ
from . import compiler as compilerOBJ
from .mesonmain import MesonMain
from .dependencyfallbacks import DependencyFallbacksHolder
from .interpreterobjects import (
from .type_checking import (
from . import primitives as P_OBJ
from pathlib import Path
from enum import Enum
import os
import shutil
import uuid
import re
import stat
import collections
import typing as T
import textwrap
import importlib
import copy
def run_command_impl(self, args: T.Tuple[T.Union[build.Executable, ExternalProgram, compilers.Compiler, mesonlib.File, str], T.List[T.Union[build.Executable, ExternalProgram, compilers.Compiler, mesonlib.File, str]]], kwargs: 'kwtypes.RunCommand', in_builddir: bool=False) -> RunProcess:
    cmd, cargs = args
    capture = kwargs['capture']
    env = kwargs['env']
    srcdir = self.environment.get_source_dir()
    builddir = self.environment.get_build_dir()
    check = kwargs['check']
    if check is None:
        mlog.warning(implicit_check_false_warning, once=True)
        check = False
    overridden_msg = 'Program {!r} was overridden with the compiled executable {!r} and therefore cannot be used during configuration'
    expanded_args: T.List[str] = []
    if isinstance(cmd, build.Executable):
        for name, exe in self.build.find_overrides.items():
            if cmd == exe:
                progname = name
                break
        else:
            raise InterpreterException(f'Program {cmd.description()!r} is a compiled executable and therefore cannot be used during configuration')
        raise InterpreterException(overridden_msg.format(progname, cmd.description()))
    if isinstance(cmd, ExternalProgram):
        if not cmd.found():
            raise InterpreterException(f'command {cmd.get_name()!r} not found or not executable')
    elif isinstance(cmd, compilers.Compiler):
        exelist = cmd.get_exelist()
        cmd = exelist[0]
        prog = ExternalProgram(cmd, silent=True)
        if not prog.found():
            raise InterpreterException(f'Program {cmd!r} not found or not executable')
        cmd = prog
        expanded_args = exelist[1:]
    else:
        if isinstance(cmd, mesonlib.File):
            cmd = cmd.absolute_path(srcdir, builddir)
        search_dir = os.path.join(srcdir, self.subdir)
        prog = ExternalProgram(cmd, silent=True, search_dir=search_dir)
        if not prog.found():
            raise InterpreterException(f'Program or command {cmd!r} not found or not executable')
        cmd = prog
    for a in cargs:
        if isinstance(a, str):
            expanded_args.append(a)
        elif isinstance(a, mesonlib.File):
            expanded_args.append(a.absolute_path(srcdir, builddir))
        elif isinstance(a, ExternalProgram):
            expanded_args.append(a.get_path())
        elif isinstance(a, compilers.Compiler):
            FeatureNew.single_use('Compiler object as a variadic argument to `run_command`', '0.61.0', self.subproject, location=self.current_node)
            prog = ExternalProgram(a.exelist[0], silent=True)
            if not prog.found():
                raise InterpreterException(f'Program {cmd!r} not found or not executable')
            expanded_args.append(prog.get_path())
        else:
            raise InterpreterException(overridden_msg.format(a.name, cmd.description()))
    self.add_build_def_file(cmd.get_path())
    for a in expanded_args:
        if not os.path.isabs(a):
            a = os.path.join(builddir if in_builddir else srcdir, self.subdir, a)
        self.add_build_def_file(a)
    return RunProcess(cmd, expanded_args, env, srcdir, builddir, self.subdir, self.environment.get_build_command() + ['introspect'], in_builddir=in_builddir, check=check, capture=capture)