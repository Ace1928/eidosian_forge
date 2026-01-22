from __future__ import annotations
import os
import typing as T
from .. import mesonlib
from .. import dependencies
from .. import build
from .. import mlog, coredata
from ..mesonlib import MachineChoice, OptionKey
from ..programs import OverrideProgram, ExternalProgram
from ..interpreter.type_checking import ENV_KW, ENV_METHOD_KW, ENV_SEPARATOR_KW, env_convertor_with_method
from ..interpreterbase import (MesonInterpreterObject, FeatureNew, FeatureDeprecated,
from .primitives import MesonVersionString
from .type_checking import NATIVE_KW, NoneType
@noArgsFlattening
@FeatureNew('meson.get_external_property', '0.54.0')
@typed_pos_args('meson.get_external_property', str, optargs=[object])
@typed_kwargs('meson.get_external_property', NATIVE_KW)
def get_external_property_method(self, args: T.Tuple[str, T.Optional[object]], kwargs: 'NativeKW') -> object:
    propname, fallback = args
    return self.__get_external_property_impl(propname, fallback, kwargs['native'])