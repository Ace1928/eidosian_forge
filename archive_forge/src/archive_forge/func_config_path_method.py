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
@noPosargs
@noKwargs
def config_path_method(self, *args: T.Any, **kwargs: T.Any) -> str:
    conf = self.held_object.hotdoc_conf.absolute_path(self.interpreter.environment.source_dir, self.interpreter.environment.build_dir)
    return conf