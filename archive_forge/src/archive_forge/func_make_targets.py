from __future__ import annotations
import os, subprocess
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build, mesonlib, mlog
from ..build import CustomTarget, CustomTargetIndex
from ..dependencies import Dependency, InternalDependency
from ..interpreterbase import (
from ..interpreter.interpreterobjects import _CustomTargetHolder
from ..interpreter.type_checking import NoneType
from ..mesonlib import File, MesonException
from ..programs import ExternalProgram
def make_targets(self) -> T.Tuple[HotdocTarget, mesonlib.ExecutableSerialisation]:
    self.check_forbidden_args()
    self.process_known_arg('--index', value_processor=self.ensure_file)
    self.process_known_arg('--project-version')
    self.process_known_arg('--sitemap', value_processor=self.ensure_file)
    self.process_known_arg('--html-extra-theme', value_processor=self.ensure_dir)
    self.include_paths.update((self.ensure_dir(v) for v in self.kwargs.pop('include_paths')))
    self.process_known_arg('--c-include-directories', argname='dependencies', value_processor=self.process_dependencies)
    self.process_gi_c_source_roots()
    self.process_extra_assets()
    self.add_extension_paths(self.kwargs.pop('extra_extension_paths'))
    self.process_subprojects()
    self.extra_depends.extend(self.kwargs.pop('depends'))
    install = self.kwargs.pop('install')
    self.process_extra_args()
    fullname = self.name + '-doc'
    hotdoc_config_name = fullname + '.json'
    hotdoc_config_path = os.path.join(self.builddir, self.subdir, hotdoc_config_name)
    with open(hotdoc_config_path, 'w', encoding='utf-8') as f:
        f.write('{}')
    self.cmd += ['--conf-file', hotdoc_config_path]
    self.include_paths.add(os.path.join(self.builddir, self.subdir))
    self.include_paths.add(os.path.join(self.sourcedir, self.subdir))
    depfile = os.path.join(self.builddir, self.subdir, self.name + '.deps')
    self.cmd += ['--deps-file-dest', depfile]
    for path in self.include_paths:
        self.cmd.extend(['--include-path', path])
    if self.state.environment.coredata.get_option(mesonlib.OptionKey('werror', subproject=self.state.subproject)):
        self.cmd.append('--fatal-warnings')
    self.generate_hotdoc_config()
    target_cmd = self.build_command + ['--internal', 'hotdoc'] + self.hotdoc.get_command() + ['run', '--conf-file', hotdoc_config_name] + ['--builddir', os.path.join(self.builddir, self.subdir)]
    target = HotdocTarget(fullname, subdir=self.subdir, subproject=self.state.subproject, environment=self.state.environment, hotdoc_conf=File.from_built_file(self.subdir, hotdoc_config_name), extra_extension_paths=self._extra_extension_paths, extra_assets=self._extra_assets, subprojects=self._subprojects, command=target_cmd, extra_depends=self.extra_depends, outputs=[fullname], sources=[], depfile=os.path.basename(depfile), build_by_default=self.build_by_default)
    install_script = None
    if install:
        datadir = os.path.join(self.state.get_option('prefix'), self.state.get_option('datadir'))
        devhelp = self.kwargs.get('devhelp_activate', False)
        if not isinstance(devhelp, bool):
            FeatureDeprecated.single_use('hotdoc.generate_doc() devhelp_activate must be boolean', '1.1.0', self.state.subproject)
            devhelp = False
        if devhelp:
            install_from = os.path.join(fullname, 'devhelp')
            install_to = os.path.join(datadir, 'devhelp')
        else:
            install_from = os.path.join(fullname, 'html')
            install_to = os.path.join(datadir, 'doc', self.name, 'html')
        install_script = self.state.backend.get_executable_serialisation(self.build_command + ['--internal', 'hotdoc', '--install', install_from, '--docdir', install_to, '--name', self.name, '--builddir', os.path.join(self.builddir, self.subdir)] + self.hotdoc.get_command() + ['run', '--conf-file', hotdoc_config_name])
        install_script.tag = 'doc'
    return (target, install_script)