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
@FeatureNew('subproject', '0.51.0')
@typed_pos_args('cmake.subproject', str)
@typed_kwargs('cmake.subproject', REQUIRED_KW, KwargInfo('options', (CMakeSubprojectOptions, NoneType), since='0.55.0'), KwargInfo('cmake_options', ContainerTypeInfo(list, str), default=[], listify=True, deprecated='0.55.0', deprecated_message='Use options instead'))
def subproject(self, state: ModuleState, args: T.Tuple[str], kwargs_: Subproject) -> T.Union[SubprojectHolder, CMakeSubproject]:
    if kwargs_['cmake_options'] and kwargs_['options'] is not None:
        raise InterpreterException('"options" cannot be used together with "cmake_options"')
    dirname = args[0]
    kw: kwargs.DoSubproject = {'required': kwargs_['required'], 'options': kwargs_['options'], 'cmake_options': kwargs_['cmake_options'], 'default_options': {}, 'version': []}
    subp = self.interpreter.do_subproject(dirname, kw, force_method='cmake')
    if not subp.found():
        return subp
    return CMakeSubproject(subp)