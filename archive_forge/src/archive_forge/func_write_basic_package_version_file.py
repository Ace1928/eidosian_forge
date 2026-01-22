from __future__ import annotations
import re
import os, os.path, pathlib
import shutil
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleObject, ModuleInfo
from .. import build, mesonlib, mlog, dependencies
from ..cmake import TargetOptions, cmake_defines_to_args
from ..interpreter import SubprojectHolder
from ..interpreter.type_checking import REQUIRED_KW, INSTALL_DIR_KW, NoneType, in_set_validator
from ..interpreterbase import (
@noPosargs
@typed_kwargs('cmake.write_basic_package_version_file', KwargInfo('arch_independent', bool, default=False, since='0.62.0'), KwargInfo('compatibility', str, default='AnyNewerVersion', validator=in_set_validator(set(COMPATIBILITIES))), KwargInfo('name', str, required=True), KwargInfo('version', str, required=True), INSTALL_DIR_KW)
def write_basic_package_version_file(self, state: ModuleState, args: TYPE_var, kwargs: 'WriteBasicPackageVersionFile') -> ModuleReturnValue:
    arch_independent = kwargs['arch_independent']
    compatibility = kwargs['compatibility']
    name = kwargs['name']
    version = kwargs['version']
    if not self.detect_cmake(state):
        raise mesonlib.MesonException('Unable to find cmake')
    pkgroot = pkgroot_name = kwargs['install_dir']
    if pkgroot is None:
        pkgroot = os.path.join(state.environment.coredata.get_option(mesonlib.OptionKey('libdir')), 'cmake', name)
        pkgroot_name = os.path.join('{libdir}', 'cmake', name)
    template_file = os.path.join(self.cmake_root, 'Modules', f'BasicConfigVersion-{compatibility}.cmake.in')
    if not os.path.exists(template_file):
        raise mesonlib.MesonException(f"your cmake installation doesn't support the {compatibility} compatibility")
    version_file = os.path.join(state.environment.scratch_dir, f'{name}ConfigVersion.cmake')
    conf: T.Dict[str, T.Union[str, bool, int]] = {'CVF_VERSION': version, 'CMAKE_SIZEOF_VOID_P': str(self.detect_voidp_size(state.environment)), 'CVF_ARCH_INDEPENDENT': arch_independent}
    mesonlib.do_conf_file(template_file, version_file, build.ConfigurationData(conf), 'meson')
    res = build.Data([mesonlib.File(True, state.environment.get_scratch_dir(), version_file)], pkgroot, pkgroot_name, None, state.subproject)
    return ModuleReturnValue(res, [res])