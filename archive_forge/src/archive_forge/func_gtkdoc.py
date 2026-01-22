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
@typed_pos_args('gnome.gtkdoc', str)
@typed_kwargs('gnome.gtkdoc', KwargInfo('c_args', ContainerTypeInfo(list, str), since='0.48.0', default=[], listify=True), KwargInfo('check', bool, default=False, since='0.52.0'), KwargInfo('content_files', ContainerTypeInfo(list, (str, mesonlib.File, GeneratedList, CustomTarget, CustomTargetIndex)), default=[], listify=True), KwargInfo('dependencies', ContainerTypeInfo(list, (Dependency, build.SharedLibrary, build.StaticLibrary)), listify=True, default=[]), KwargInfo('expand_content_files', ContainerTypeInfo(list, (str, mesonlib.File)), default=[], listify=True), KwargInfo('fixxref_args', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('gobject_typesfile', ContainerTypeInfo(list, (str, mesonlib.File)), default=[], listify=True), KwargInfo('html_args', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('html_assets', ContainerTypeInfo(list, (str, mesonlib.File)), default=[], listify=True), KwargInfo('ignore_headers', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('include_directories', ContainerTypeInfo(list, (str, build.IncludeDirs)), listify=True, default=[]), KwargInfo('install', bool, default=True), KwargInfo('install_dir', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('main_sgml', (str, NoneType)), KwargInfo('main_xml', (str, NoneType)), KwargInfo('mkdb_args', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('mode', str, default='auto', since='0.37.0', validator=in_set_validator({'xml', 'sgml', 'none', 'auto'})), KwargInfo('module_version', str, default='', since='0.48.0'), KwargInfo('namespace', str, default='', since='0.37.0'), KwargInfo('scan_args', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('scanobjs_args', ContainerTypeInfo(list, str), default=[], listify=True), KwargInfo('src_dir', ContainerTypeInfo(list, (str, build.IncludeDirs)), listify=True, required=True))
def gtkdoc(self, state: 'ModuleState', args: T.Tuple[str], kwargs: 'GtkDoc') -> ModuleReturnValue:
    modulename = args[0]
    main_file = kwargs['main_sgml']
    main_xml = kwargs['main_xml']
    if main_xml is not None:
        if main_file is not None:
            raise InvalidArguments('gnome.gtkdoc: main_xml and main_sgml are exclusive arguments')
        main_file = main_xml
    moduleversion = kwargs['module_version']
    targetname = modulename + ('-' + moduleversion if moduleversion else '') + '-doc'
    command = state.environment.get_build_command()
    namespace = kwargs['namespace']
    state.add_language('c', MachineChoice.HOST)

    def abs_filenames(files: T.Iterable['FileOrString']) -> T.Iterator[str]:
        for f in files:
            if isinstance(f, mesonlib.File):
                yield f.absolute_path(state.environment.get_source_dir(), state.environment.get_build_dir())
            else:
                yield os.path.join(state.environment.get_source_dir(), state.subdir, f)
    src_dirs = kwargs['src_dir']
    header_dirs: T.List[str] = []
    for src_dir in src_dirs:
        if isinstance(src_dir, build.IncludeDirs):
            header_dirs.extend(src_dir.to_string_list(state.environment.get_source_dir(), state.environment.get_build_dir()))
        else:
            header_dirs.append(src_dir)
    t_args: T.List[str] = ['--internal', 'gtkdoc', '--sourcedir=' + state.environment.get_source_dir(), '--builddir=' + state.environment.get_build_dir(), '--subdir=' + state.subdir, '--headerdirs=' + '@@'.join(header_dirs), '--mainfile=' + main_file, '--modulename=' + modulename, '--moduleversion=' + moduleversion, '--mode=' + kwargs['mode']]
    for tool in ['scan', 'scangobj', 'mkdb', 'mkhtml', 'fixxref']:
        program_name = 'gtkdoc-' + tool
        program = state.find_program(program_name)
        path = program.get_path()
        assert path is not None, "This shouldn't be possible since program should be found"
        t_args.append(f'--{program_name}={path}')
    if namespace:
        t_args.append('--namespace=' + namespace)
    exe_wrapper = state.environment.get_exe_wrapper()
    if exe_wrapper:
        t_args.append('--run=' + ' '.join(exe_wrapper.get_command()))
    t_args.append(f'--htmlargs={'@@'.join(kwargs['html_args'])}')
    t_args.append(f'--scanargs={'@@'.join(kwargs['scan_args'])}')
    t_args.append(f'--scanobjsargs={'@@'.join(kwargs['scanobjs_args'])}')
    t_args.append(f'--gobjects-types-file={'@@'.join(abs_filenames(kwargs['gobject_typesfile']))}')
    t_args.append(f'--fixxrefargs={'@@'.join(kwargs['fixxref_args'])}')
    t_args.append(f'--mkdbargs={'@@'.join(kwargs['mkdb_args'])}')
    t_args.append(f'--html-assets={'@@'.join(abs_filenames(kwargs['html_assets']))}')
    depends: T.List['build.GeneratedTypes'] = []
    content_files = []
    for s in kwargs['content_files']:
        if isinstance(s, (CustomTarget, CustomTargetIndex)):
            depends.append(s)
            for o in s.get_outputs():
                content_files.append(os.path.join(state.environment.get_build_dir(), state.backend.get_target_dir(s), o))
        elif isinstance(s, mesonlib.File):
            content_files.append(s.absolute_path(state.environment.get_source_dir(), state.environment.get_build_dir()))
        elif isinstance(s, GeneratedList):
            depends.append(s)
            for gen_src in s.get_outputs():
                content_files.append(os.path.join(state.environment.get_source_dir(), state.subdir, gen_src))
        else:
            content_files.append(os.path.join(state.environment.get_source_dir(), state.subdir, s))
    t_args += ['--content-files=' + '@@'.join(content_files)]
    t_args.append(f'--expand-content-files={'@@'.join(abs_filenames(kwargs['expand_content_files']))}')
    t_args.append(f'--ignore-headers={'@@'.join(kwargs['ignore_headers'])}')
    t_args.append(f'--installdir={'@@'.join(kwargs['install_dir'])}')
    build_args, new_depends = self._get_build_args(kwargs['c_args'], kwargs['include_directories'], kwargs['dependencies'], state, depends)
    t_args.extend(build_args)
    new_depends.extend(depends)
    custom_target = CustomTarget(targetname, state.subdir, state.subproject, state.environment, command + t_args, [], [f'{modulename}-decl.txt'], build_always_stale=True, extra_depends=new_depends, description='Generating gtkdoc {}')
    alias_target = build.AliasTarget(targetname, [custom_target], state.subdir, state.subproject, state.environment)
    if kwargs['check']:
        check_cmd = state.find_program('gtkdoc-check')
        check_env = ['DOC_MODULE=' + modulename, 'DOC_MAIN_SGML_FILE=' + main_file]
        check_args = (targetname + '-check', check_cmd)
        check_workdir = os.path.join(state.environment.get_build_dir(), state.subdir)
        state.test(check_args, env=check_env, workdir=check_workdir, depends=[custom_target])
    res: T.List[T.Union[build.Target, mesonlib.ExecutableSerialisation]] = [custom_target, alias_target]
    if kwargs['install']:
        res.append(state.backend.get_executable_serialisation(command + t_args, tag='doc'))
    return ModuleReturnValue(custom_target, res)