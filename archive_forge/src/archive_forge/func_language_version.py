from __future__ import annotations
import sysconfig
import typing as T
from .. import mesonlib
from . import ExtensionModule, ModuleInfo, ModuleState
from ..build import (
from ..interpreter.type_checking import SHARED_MOD_KWS
from ..interpreterbase import typed_kwargs, typed_pos_args, noPosargs, noKwargs, permittedKwargs
from ..programs import ExternalProgram
@noPosargs
@noKwargs
def language_version(self, state, args, kwargs):
    return sysconfig.get_python_version()