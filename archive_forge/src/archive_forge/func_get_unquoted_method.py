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
@FeatureNew('configuration_data.get_unquoted()', '0.44.0')
@typed_pos_args('configuration_data.get_unquoted', str, optargs=[(str, int, bool)])
@noKwargs
def get_unquoted_method(self, args: T.Tuple[str, T.Optional[T.Union[str, int, bool]]], kwargs: TYPE_kwargs) -> T.Union[str, int, bool]:
    name = args[0]
    if name in self.held_object:
        val = self.held_object.get(name)[0]
    elif args[1] is not None:
        val = args[1]
    else:
        raise InterpreterException(f'Entry {name} not in configuration data.')
    if isinstance(val, str) and val[0] == '"' and (val[-1] == '"'):
        return val[1:-1]
    return val