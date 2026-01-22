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
@noPosargs
@typed_kwargs('configure_file', DEPFILE_KW.evolve(since='0.52.0'), INSTALL_MODE_KW.evolve(since='0.47.0,'), INSTALL_TAG_KW.evolve(since='0.60.0'), KwargInfo('capture', bool, default=False, since='0.41.0'), KwargInfo('command', (ContainerTypeInfo(list, (build.Executable, ExternalProgram, compilers.Compiler, mesonlib.File, str), allow_empty=False), NoneType), listify=True), KwargInfo('configuration', (ContainerTypeInfo(dict, (str, int, bool)), build.ConfigurationData, NoneType)), KwargInfo('copy', bool, default=False, since='0.47.0', deprecated='0.64.0', deprecated_message='Use fs.copyfile instead'), KwargInfo('encoding', str, default='utf-8', since='0.47.0'), KwargInfo('format', str, default='meson', since='0.46.0', validator=in_set_validator({'meson', 'cmake', 'cmake@'})), KwargInfo('input', ContainerTypeInfo(list, (mesonlib.File, str)), listify=True, default=[]), KwargInfo('install', (bool, NoneType), since='0.50.0'), KwargInfo('install_dir', (str, bool), default='', validator=lambda x: 'must be `false` if boolean' if x is True else None), OUTPUT_KW, KwargInfo('output_format', str, default='c', since='0.47.0', since_values={'json': '1.3.0'}, validator=in_set_validator({'c', 'json', 'nasm'})), KwargInfo('macro_name', (str, NoneType), default=None, since='1.3.0'))
def func_configure_file(self, node: mparser.BaseNode, args: T.List[TYPE_var], kwargs: kwtypes.ConfigureFile):
    actions = sorted((x for x in ['configuration', 'command', 'copy'] if kwargs[x] not in [None, False]))
    num_actions = len(actions)
    if num_actions == 0:
        raise InterpreterException("Must specify an action with one of these keyword arguments: 'configuration', 'command', or 'copy'.")
    elif num_actions == 2:
        raise InterpreterException('Must not specify both {!r} and {!r} keyword arguments since they are mutually exclusive.'.format(*actions))
    elif num_actions == 3:
        raise InterpreterException('Must specify one of {!r}, {!r}, and {!r} keyword arguments since they are mutually exclusive.'.format(*actions))
    if kwargs['capture'] and (not kwargs['command']):
        raise InvalidArguments('configure_file: "capture" keyword requires "command" keyword.')
    install_mode = self._warn_kwarg_install_mode_sticky(kwargs['install_mode'])
    fmt = kwargs['format']
    output_format = kwargs['output_format']
    depfile = kwargs['depfile']
    inputs = self.source_strings_to_files(kwargs['input'])
    inputs_abs = []
    for f in inputs:
        if isinstance(f, mesonlib.File):
            inputs_abs.append(f.absolute_path(self.environment.source_dir, self.environment.build_dir))
            self.add_build_def_file(f)
        else:
            raise InterpreterException('Inputs can only be strings or file objects')
    output = kwargs['output']
    if inputs_abs:
        values = mesonlib.get_filenames_templates_dict(inputs_abs, None)
        outputs = mesonlib.substitute_values([output], values)
        output = outputs[0]
        if depfile:
            depfile = mesonlib.substitute_values([depfile], values)[0]
    ofile_rpath = os.path.join(self.subdir, output)
    if ofile_rpath in self.configure_file_outputs:
        mesonbuildfile = os.path.join(self.subdir, 'meson.build')
        current_call = f'{mesonbuildfile}:{self.current_lineno}'
        first_call = '{}:{}'.format(mesonbuildfile, self.configure_file_outputs[ofile_rpath])
        mlog.warning('Output file', mlog.bold(ofile_rpath, True), 'for configure_file() at', current_call, 'overwrites configure_file() output at', first_call)
    else:
        self.configure_file_outputs[ofile_rpath] = self.current_lineno
    ofile_path, ofile_fname = os.path.split(os.path.join(self.subdir, output))
    ofile_abs = os.path.join(self.environment.build_dir, ofile_path, ofile_fname)
    if kwargs['configuration'] is not None:
        conf = kwargs['configuration']
        if isinstance(conf, dict):
            FeatureNew.single_use('configure_file.configuration dictionary', '0.49.0', self.subproject, location=node)
            for k, v in conf.items():
                if not isinstance(v, (str, int, bool)):
                    raise InvalidArguments(f'"configuration_data": initial value dictionary key "{k!r}"" must be "str | int | bool", not "{v!r}"')
            conf = build.ConfigurationData(conf)
        mlog.log('Configuring', mlog.bold(output), 'using configuration')
        if len(inputs) > 1:
            raise InterpreterException('At most one input file can given in configuration mode')
        if inputs:
            os.makedirs(os.path.join(self.environment.build_dir, self.subdir), exist_ok=True)
            file_encoding = kwargs['encoding']
            missing_variables, confdata_useless = mesonlib.do_conf_file(inputs_abs[0], ofile_abs, conf, fmt, file_encoding, self.subproject)
            if missing_variables:
                var_list = ', '.join((repr(m) for m in sorted(missing_variables)))
                mlog.warning(f"The variable(s) {var_list} in the input file '{inputs[0]}' are not present in the given configuration data.", location=node)
            if confdata_useless:
                ifbase = os.path.basename(inputs_abs[0])
                tv = FeatureNew.get_target_version(self.subproject)
                if FeatureNew.check_version(tv, '0.47.0'):
                    mlog.warning(f"Got an empty configuration_data() object and found no substitutions in the input file {ifbase!r}. If you want to copy a file to the build dir, use the 'copy:' keyword argument added in 0.47.0", location=node)
        else:
            macro_name = kwargs['macro_name']
            mesonlib.dump_conf_header(ofile_abs, conf, output_format, macro_name)
        conf.used = True
    elif kwargs['command'] is not None:
        if len(inputs) > 1:
            FeatureNew.single_use('multiple inputs in configure_file()', '0.52.0', self.subproject, location=node)
        values = mesonlib.get_filenames_templates_dict(inputs_abs, [ofile_abs])
        if depfile:
            depfile = os.path.join(self.environment.get_scratch_dir(), depfile)
            values['@DEPFILE@'] = depfile
        _cmd = mesonlib.substitute_values(kwargs['command'], values)
        mlog.log('Configuring', mlog.bold(output), 'with command')
        cmd, *args = _cmd
        res = self.run_command_impl((cmd, args), {'capture': True, 'check': True, 'env': EnvironmentVariables()}, True)
        if kwargs['capture']:
            dst_tmp = ofile_abs + '~'
            file_encoding = kwargs['encoding']
            with open(dst_tmp, 'w', encoding=file_encoding) as f:
                f.writelines(res.stdout)
            if inputs_abs:
                shutil.copymode(inputs_abs[0], dst_tmp)
            mesonlib.replace_if_different(ofile_abs, dst_tmp)
        if depfile:
            mlog.log('Reading depfile:', mlog.bold(depfile))
            with open(depfile, encoding='utf-8') as f:
                df = DepFile(f.readlines())
                deps = df.get_all_dependencies(ofile_fname)
                for dep in deps:
                    self.add_build_def_file(dep)
    elif kwargs['copy']:
        if len(inputs_abs) != 1:
            raise InterpreterException('Exactly one input file must be given in copy mode')
        os.makedirs(os.path.join(self.environment.build_dir, self.subdir), exist_ok=True)
        shutil.copy2(inputs_abs[0], ofile_abs)
    idir = kwargs['install_dir']
    if idir is False:
        idir = ''
        FeatureDeprecated.single_use('configure_file install_dir: false', '0.50.0', self.subproject, 'Use the `install:` kwarg instead', location=node)
    install = kwargs['install'] if kwargs['install'] is not None else idir != ''
    if install:
        if not idir:
            raise InterpreterException('"install_dir" must be specified when "install" in a configure_file is true')
        idir_name = idir
        if isinstance(idir_name, P_OBJ.OptionString):
            idir_name = idir_name.optname
        cfile = mesonlib.File.from_built_file(ofile_path, ofile_fname)
        install_tag = kwargs['install_tag']
        self.build.data.append(build.Data([cfile], idir, idir_name, install_mode, self.subproject, install_tag=install_tag, data_type='configure'))
    return mesonlib.File.from_built_file(self.subdir, output)