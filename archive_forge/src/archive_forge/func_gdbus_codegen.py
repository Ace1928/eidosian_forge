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
@typed_pos_args('gnome.gdbus_codegen', str, optargs=[(str, mesonlib.File, CustomTarget, CustomTargetIndex, GeneratedList)])
@typed_kwargs('gnome.gdbus_codegen', _BUILD_BY_DEFAULT.evolve(since='0.40.0'), DEPENDENCY_SOURCES_KW.evolve(since='0.46.0'), KwargInfo('extra_args', ContainerTypeInfo(list, str), since='0.47.0', default=[], listify=True), KwargInfo('interface_prefix', (str, NoneType)), KwargInfo('namespace', (str, NoneType)), KwargInfo('object_manager', bool, default=False), KwargInfo('annotations', ContainerTypeInfo(list, (list, str)), default=[], validator=annotations_validator, convertor=lambda x: [x] if x and isinstance(x[0], str) else x), KwargInfo('install_header', bool, default=False, since='0.46.0'), KwargInfo('docbook', (str, NoneType)), KwargInfo('autocleanup', str, default='default', since='0.47.0', validator=in_set_validator({'all', 'none', 'objects'})), INSTALL_DIR_KW.evolve(since='0.46.0'))
def gdbus_codegen(self, state: 'ModuleState', args: T.Tuple[str, T.Optional[T.Union['FileOrString', build.GeneratedTypes]]], kwargs: 'GdbusCodegen') -> ModuleReturnValue:
    namebase = args[0]
    xml_files: T.List[T.Union['FileOrString', build.GeneratedTypes]] = [args[1]] if args[1] else []
    cmd: T.List[T.Union['ToolType', str]] = [self._find_tool(state, 'gdbus-codegen')]
    cmd.extend(kwargs['extra_args'])
    glib_version = self._get_native_glib_version(state)
    if not mesonlib.version_compare(glib_version, '>= 2.49.1'):
        if kwargs['autocleanup'] != 'default':
            mlog.warning(f"Glib version ({glib_version}) is too old to support the 'autocleanup' kwarg, need 2.49.1 or newer")
    else:
        ac = kwargs['autocleanup']
        if ac == 'default':
            ac = 'all'
        cmd.extend(['--c-generate-autocleanup', ac])
    if kwargs['interface_prefix'] is not None:
        cmd.extend(['--interface-prefix', kwargs['interface_prefix']])
    if kwargs['namespace'] is not None:
        cmd.extend(['--c-namespace', kwargs['namespace']])
    if kwargs['object_manager']:
        cmd.extend(['--c-generate-object-manager'])
    xml_files.extend(kwargs['sources'])
    build_by_default = kwargs['build_by_default']
    for annot in kwargs['annotations']:
        cmd.append('--annotate')
        cmd.extend(annot)
    targets = []
    install_header = kwargs['install_header']
    install_dir = kwargs['install_dir'] or state.environment.coredata.get_option(mesonlib.OptionKey('includedir'))
    assert isinstance(install_dir, str), 'for mypy'
    output = namebase + '.c'
    if mesonlib.version_compare(glib_version, '>= 2.56.2'):
        c_cmd = cmd + ['--body', '--output', '@OUTPUT@', '@INPUT@']
    else:
        if kwargs['docbook'] is not None:
            docbook = kwargs['docbook']
            cmd += ['--generate-docbook', docbook]
        if mesonlib.version_compare(glib_version, '>= 2.51.3'):
            cmd += ['--output-directory', '@OUTDIR@', '--generate-c-code', namebase, '@INPUT@']
        else:
            self._print_gdbus_warning()
            cmd += ['--generate-c-code', '@OUTDIR@/' + namebase, '@INPUT@']
        c_cmd = cmd
    cfile_custom_target = CustomTarget(output, state.subdir, state.subproject, state.environment, c_cmd, xml_files, [output], build_by_default=build_by_default, description='Generating gdbus source {}')
    targets.append(cfile_custom_target)
    output = namebase + '.h'
    if mesonlib.version_compare(glib_version, '>= 2.56.2'):
        hfile_cmd = cmd + ['--header', '--output', '@OUTPUT@', '@INPUT@']
        depends = []
    else:
        hfile_cmd = cmd
        depends = [cfile_custom_target]
    hfile_custom_target = CustomTarget(output, state.subdir, state.subproject, state.environment, hfile_cmd, xml_files, [output], build_by_default=build_by_default, extra_depends=depends, install=install_header, install_dir=[install_dir], install_tag=['devel'], description='Generating gdbus header {}')
    targets.append(hfile_custom_target)
    if kwargs['docbook'] is not None:
        docbook = kwargs['docbook']
        output = namebase + '-docbook'
        outputs = []
        for f in xml_files:
            outputs.append('{}-{}'.format(docbook, os.path.basename(str(f))))
        if mesonlib.version_compare(glib_version, '>= 2.56.2'):
            docbook_cmd = cmd + ['--output-directory', '@OUTDIR@', '--generate-docbook', docbook, '@INPUT@']
            depends = []
        else:
            docbook_cmd = cmd
            depends = [cfile_custom_target]
        docbook_custom_target = CustomTarget(output, state.subdir, state.subproject, state.environment, docbook_cmd, xml_files, outputs, build_by_default=build_by_default, extra_depends=depends, description='Generating gdbus docbook {}')
        targets.append(docbook_custom_target)
    return ModuleReturnValue(targets, targets)