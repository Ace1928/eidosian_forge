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
def check_extra_arg_type(self, arg: str, value: TYPE_var) -> None:
    if isinstance(value, list):
        for v in value:
            self.check_extra_arg_type(arg, v)
        return
    valid_types = (str, bool, File, build.IncludeDirs, CustomTarget, CustomTargetIndex, build.BuildTarget)
    if not isinstance(value, valid_types):
        raise InvalidArguments('Argument "{}={}" should be of type: {}.'.format(arg, value, [t.__name__ for t in valid_types]))