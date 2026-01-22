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
@typed_pos_args('gnome.genmarshal', str)
@typed_kwargs('gnome.genmarshal', DEPEND_FILES_KW.evolve(since='0.61.0'), DEPENDS_KW.evolve(since='0.61.0'), INSTALL_KW.evolve(name='install_header'), INSTALL_DIR_KW, KwargInfo('extra_args', ContainerTypeInfo(list, str), listify=True, default=[]), KwargInfo('internal', bool, default=False), KwargInfo('nostdinc', bool, default=False), KwargInfo('prefix', (str, NoneType)), KwargInfo('skip_source', bool, default=False), KwargInfo('sources', ContainerTypeInfo(list, (str, mesonlib.File), allow_empty=False), listify=True, required=True), KwargInfo('stdinc', bool, default=False), KwargInfo('valist_marshallers', bool, default=False))
def genmarshal(self, state: 'ModuleState', args: T.Tuple[str], kwargs: 'GenMarshal') -> ModuleReturnValue:
    output = args[0]
    sources = kwargs['sources']
    new_genmarshal = mesonlib.version_compare(self._get_native_glib_version(state), '>= 2.53.3')
    cmd: T.List[T.Union['ToolType', str]] = [self._find_tool(state, 'glib-genmarshal')]
    if kwargs['prefix']:
        cmd.extend(['--prefix', kwargs['prefix']])
    if kwargs['extra_args']:
        if new_genmarshal:
            cmd.extend(kwargs['extra_args'])
        else:
            mlog.warning('The current version of GLib does not support extra arguments \nfor glib-genmarshal. You need at least GLib 2.53.3. See ', mlog.bold('https://github.com/mesonbuild/meson/pull/2049'), once=True, fatal=False)
    for k in ['internal', 'nostdinc', 'skip_source', 'stdinc', 'valist_marshallers']:
        if kwargs[k]:
            cmd.append(f'--{k.replace('_', '-')}')
    install_header = kwargs['install_header']
    capture = False
    if mesonlib.version_compare(self._get_native_glib_version(state), '>= 2.51.0'):
        cmd += ['--output', '@OUTPUT@']
    else:
        capture = True
    header_file = output + '.h'
    h_cmd = cmd + ['--header', '@INPUT@']
    if new_genmarshal:
        h_cmd += ['--pragma-once']
    header = CustomTarget(output + '_h', state.subdir, state.subproject, state.environment, h_cmd, sources, [header_file], install=install_header, install_dir=[kwargs['install_dir']] if kwargs['install_dir'] else [], install_tag=['devel'], capture=capture, depend_files=kwargs['depend_files'], description='Generating glib marshaller header {}')
    c_cmd = cmd + ['--body', '@INPUT@']
    extra_deps: T.List[CustomTarget] = []
    if mesonlib.version_compare(self._get_native_glib_version(state), '>= 2.53.4'):
        c_cmd += ['--include-header', header_file]
        extra_deps.append(header)
    body = CustomTarget(output + '_c', state.subdir, state.subproject, state.environment, c_cmd, sources, [f'{output}.c'], capture=capture, depend_files=kwargs['depend_files'], extra_depends=extra_deps, description='Generating glib marshaller source {}')
    rv = [body, header]
    return ModuleReturnValue(rv, rv)