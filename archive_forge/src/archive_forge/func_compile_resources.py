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
@typed_pos_args('gnome.compile_resources', str, (str, mesonlib.File, CustomTarget, CustomTargetIndex, GeneratedList))
@typed_kwargs('gnome.compile_resources', _BUILD_BY_DEFAULT, _EXTRA_ARGS_KW, INSTALL_KW, INSTALL_KW.evolve(name='install_header', since='0.37.0'), INSTALL_DIR_KW, KwargInfo('c_name', (str, NoneType)), KwargInfo('dependencies', ContainerTypeInfo(list, (mesonlib.File, CustomTarget, CustomTargetIndex)), default=[], listify=True), KwargInfo('export', bool, default=False, since='0.37.0'), KwargInfo('gresource_bundle', bool, default=False, since='0.37.0'), KwargInfo('source_dir', ContainerTypeInfo(list, str), default=[], listify=True))
def compile_resources(self, state: 'ModuleState', args: T.Tuple[str, 'FileOrString'], kwargs: 'CompileResources') -> 'ModuleReturnValue':
    self.__print_gresources_warning(state)
    glib_version = self._get_native_glib_version(state)
    glib_compile_resources = self._find_tool(state, 'glib-compile-resources')
    cmd: T.List[T.Union['ToolType', str]] = [glib_compile_resources, '@INPUT@']
    source_dirs = kwargs['source_dir']
    dependencies = kwargs['dependencies']
    target_name, input_file = args
    subdirs: T.List[str] = []
    depends: T.List[T.Union[CustomTarget, CustomTargetIndex]] = []
    for dep in dependencies:
        if isinstance(dep, mesonlib.File):
            subdirs.append(dep.subdir)
        else:
            depends.append(dep)
            subdirs.append(dep.get_subdir())
            if not mesonlib.version_compare(glib_version, gresource_dep_needed_version):
                m = 'The "dependencies" argument of gnome.compile_resources() cannot\nbe used with the current version of glib-compile-resources due to\n<https://bugzilla.gnome.org/show_bug.cgi?id=774368>'
                raise MesonException(m)
    if not mesonlib.version_compare(glib_version, gresource_dep_needed_version):
        if isinstance(input_file, mesonlib.File):
            if input_file.is_built:
                ifile = os.path.join(state.environment.get_build_dir(), input_file.subdir, input_file.fname)
            else:
                ifile = os.path.join(input_file.subdir, input_file.fname)
        elif isinstance(input_file, (CustomTarget, CustomTargetIndex, GeneratedList)):
            raise MesonException('Resource xml files generated at build-time cannot be used with gnome.compile_resources() in the current version of glib-compile-resources because we need to scan the xml for dependencies due to <https://bugzilla.gnome.org/show_bug.cgi?id=774368>\nUse configure_file() instead to generate it at configure-time.')
        else:
            ifile = os.path.join(state.subdir, input_file)
        depend_files, depends, subdirs = self._get_gresource_dependencies(state, ifile, source_dirs, dependencies)
    source_dirs = [os.path.join(state.build_to_src, state.subdir, d) for d in source_dirs]
    source_dirs += subdirs
    source_dirs.append(os.path.join(state.build_to_src, state.subdir))
    source_dirs = list(OrderedSet((os.path.normpath(dir) for dir in source_dirs)))
    for source_dir in source_dirs:
        cmd += ['--sourcedir', source_dir]
    if kwargs['c_name']:
        cmd += ['--c-name', kwargs['c_name']]
    if not kwargs['export']:
        cmd += ['--internal']
    cmd += ['--generate', '--target', '@OUTPUT@']
    cmd += kwargs['extra_args']
    gresource = kwargs['gresource_bundle']
    if gresource:
        output = f'{target_name}.gresource'
        name = f'{target_name}_gresource'
    elif 'c' in state.environment.coredata.compilers.host:
        output = f'{target_name}.c'
        name = f'{target_name}_c'
    elif 'cpp' in state.environment.coredata.compilers.host:
        output = f'{target_name}.cpp'
        name = f'{target_name}_cpp'
    else:
        raise MesonException('Compiling GResources into code is only supported in C and C++ projects')
    if kwargs['install'] and (not gresource):
        raise MesonException('The install kwarg only applies to gresource bundles, see install_header')
    install_header = kwargs['install_header']
    if install_header and gresource:
        raise MesonException('The install_header kwarg does not apply to gresource bundles')
    if install_header and (not kwargs['export']):
        raise MesonException('GResource header is installed yet export is not enabled')
    depfile: T.Optional[str] = None
    target_cmd: T.List[T.Union['ToolType', str]]
    if not mesonlib.version_compare(glib_version, gresource_dep_needed_version):
        target_cmd = cmd
    else:
        depfile = f'{output}.d'
        depend_files = []
        target_cmd = copy.copy(cmd) + ['--dependency-file', '@DEPFILE@']
    target_c = GResourceTarget(name, state.subdir, state.subproject, state.environment, target_cmd, [input_file], [output], build_by_default=kwargs['build_by_default'], depfile=depfile, depend_files=depend_files, extra_depends=depends, install=kwargs['install'], install_dir=[kwargs['install_dir']] if kwargs['install_dir'] else [], install_tag=['runtime'])
    target_c.source_dirs = source_dirs
    if gresource:
        return ModuleReturnValue(target_c, [target_c])
    install_dir = kwargs['install_dir'] or state.environment.coredata.get_option(mesonlib.OptionKey('includedir'))
    assert isinstance(install_dir, str), 'for mypy'
    target_h = GResourceHeaderTarget(f'{target_name}_h', state.subdir, state.subproject, state.environment, cmd, [input_file], [f'{target_name}.h'], build_by_default=kwargs['build_by_default'], extra_depends=depends, install=install_header, install_dir=[install_dir], install_tag=['devel'])
    rv = [target_c, target_h]
    return ModuleReturnValue(rv, rv)