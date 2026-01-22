from __future__ import annotations
import copy
import itertools
import functools
import os
import subprocess
import textwrap
import typing as T
from . import (
from .. import build
from .. import interpreter
from .. import mesonlib
from .. import mlog
from ..build import CustomTarget, CustomTargetIndex, Executable, GeneratedList, InvalidArguments
from ..dependencies import Dependency, InternalDependency
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import DEPENDS_KW, DEPEND_FILES_KW, ENV_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, DEPENDENCY_SOURCES_KW, in_set_validator
from ..interpreterbase import noPosargs, noKwargs, FeatureNew, FeatureDeprecated
from ..interpreterbase import typed_kwargs, KwargInfo, ContainerTypeInfo
from ..interpreterbase.decorators import typed_pos_args
from ..mesonlib import (
from ..programs import OverrideProgram
from ..scripts.gettext import read_linguas
@noPosargs
@typed_kwargs('gnome.compile_schemas', _BUILD_BY_DEFAULT.evolve(since='0.40.0'), DEPEND_FILES_KW)
def compile_schemas(self, state: 'ModuleState', args: T.List['TYPE_var'], kwargs: 'CompileSchemas') -> ModuleReturnValue:
    srcdir = os.path.join(state.build_to_src, state.subdir)
    outdir = state.subdir
    cmd: T.List[T.Union['ToolType', str]] = [self._find_tool(state, 'glib-compile-schemas'), '--targetdir', outdir, srcdir]
    if state.subdir == '':
        targetname = 'gsettings-compile'
    else:
        targetname = 'gsettings-compile-' + state.subdir.replace('/', '_')
    target_g = CustomTarget(targetname, state.subdir, state.subproject, state.environment, cmd, [], ['gschemas.compiled'], build_by_default=kwargs['build_by_default'], depend_files=kwargs['depend_files'], description='Compiling gschemas {}')
    self._devenv_prepend('GSETTINGS_SCHEMA_DIR', os.path.join(state.environment.get_build_dir(), state.subdir))
    return ModuleReturnValue(target_g, [target_g])