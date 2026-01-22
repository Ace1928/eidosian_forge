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
@FeatureNew('dependency.get_configtool_variable', '0.44.0')
@FeatureDeprecated('dependency.get_configtool_variable', '0.56.0', 'use dependency.get_variable(configtool : ...) instead')
@noKwargs
@typed_pos_args('dependency.get_config_tool_variable', str)
def configtool_method(self, args: T.Tuple[str], kwargs: TYPE_kwargs) -> str:
    from ..dependencies.configtool import ConfigToolDependency
    if not isinstance(self.held_object, ConfigToolDependency):
        raise InvalidArguments(f'{self.held_object.get_name()!r} is not a config-tool dependency')
    return self.held_object.get_variable(configtool=args[0], default_value='')