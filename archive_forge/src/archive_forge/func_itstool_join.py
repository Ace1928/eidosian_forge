from __future__ import annotations
from os import path
import shlex
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build
from .. import mesonlib
from .. import mlog
from ..interpreter.type_checking import CT_BUILD_BY_DEFAULT, CT_INPUT_KW, INSTALL_TAG_KW, OUTPUT_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, in_set_validator
from ..interpreterbase import FeatureNew, InvalidArguments
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, noPosargs, typed_kwargs, typed_pos_args
from ..programs import ExternalProgram
from ..scripts.gettext import read_linguas
@FeatureNew('i18n.itstool_join', '0.62.0')
@noPosargs
@typed_kwargs('i18n.itstool_join', CT_BUILD_BY_DEFAULT, CT_INPUT_KW, KwargInfo('install_dir', (str, NoneType)), INSTALL_TAG_KW, OUTPUT_KW, INSTALL_KW, _ARGS.evolve(), KwargInfo('its_files', ContainerTypeInfo(list, str)), KwargInfo('mo_targets', ContainerTypeInfo(list, build.CustomTarget), required=True))
def itstool_join(self, state: 'ModuleState', args: T.List['TYPE_var'], kwargs: 'ItsJoinFile') -> ModuleReturnValue:
    if kwargs['install'] and (not kwargs['install_dir']):
        raise InvalidArguments('i18n.itstool_join: "install_dir" keyword argument must be set when "install" is true.')
    if self.tools['itstool'] is None:
        self.tools['itstool'] = state.find_program('itstool', for_machine=mesonlib.MachineChoice.BUILD)
    mo_targets = kwargs['mo_targets']
    its_files = kwargs.get('its_files', [])
    mo_fnames = []
    for target in mo_targets:
        mo_fnames.append(path.join(target.get_subdir(), target.get_outputs()[0]))
    command: T.List[T.Union[str, build.BuildTarget, build.CustomTarget, build.CustomTargetIndex, 'ExternalProgram', mesonlib.File]] = []
    command.extend(state.environment.get_build_command())
    itstool_cmd = self.tools['itstool'].get_command()
    command.extend(['--internal', 'itstool', 'join', '-i', '@INPUT@', '-o', '@OUTPUT@', '--itstool=' + ' '.join((shlex.quote(c) for c in itstool_cmd))])
    if its_files:
        for fname in its_files:
            if not path.isabs(fname):
                fname = path.join(state.environment.source_dir, state.subdir, fname)
            command.extend(['--its', fname])
    command.extend(mo_fnames)
    build_by_default = kwargs['build_by_default']
    if build_by_default is None:
        build_by_default = kwargs['install']
    install_tag = [kwargs['install_tag']] if kwargs['install_tag'] is not None else None
    ct = build.CustomTarget('', state.subdir, state.subproject, state.environment, command, kwargs['input'], [kwargs['output']], build_by_default=build_by_default, extra_depends=mo_targets, install=kwargs['install'], install_dir=[kwargs['install_dir']] if kwargs['install_dir'] is not None else None, install_tag=install_tag, description='Merging translations for {}')
    return ModuleReturnValue(ct, [ct])