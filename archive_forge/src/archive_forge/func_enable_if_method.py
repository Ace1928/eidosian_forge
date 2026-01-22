from __future__ import annotations
import os
import shlex
import subprocess
import copy
import textwrap
from pathlib import Path, PurePath
from .. import mesonlib
from .. import coredata
from .. import build
from .. import mlog
from ..modules import ModuleReturnValue, ModuleObject, ModuleState, ExtensionModule
from ..backend.backends import TestProtocol
from ..interpreterbase import (
from ..interpreter.type_checking import NoneType, ENV_KW, ENV_SEPARATOR_KW, PKGCONFIG_DEFINE_KW
from ..dependencies import Dependency, ExternalLibrary, InternalDependency
from ..programs import ExternalProgram
from ..mesonlib import HoldableObject, OptionKey, listify, Popen_safe
import typing as T
@FeatureNew('feature_option.enable_if()', '1.1.0')
@typed_pos_args('feature_option.enable_if', bool)
@typed_kwargs('feature_option.enable_if', _ERROR_MSG_KW)
def enable_if_method(self, args: T.Tuple[bool], kwargs: 'kwargs.FeatureOptionRequire') -> coredata.UserFeatureOption:
    if not args[0]:
        return copy.deepcopy(self.held_object)
    if self.value == 'disabled':
        err_msg = f'Feature {self.held_object.name} cannot be disabled'
        if kwargs['error_message']:
            err_msg += f': {kwargs['error_message']}'
        raise InterpreterException(err_msg)
    return self.as_enabled()