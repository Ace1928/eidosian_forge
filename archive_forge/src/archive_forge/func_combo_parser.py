from __future__ import annotations
import re
import typing as T
from . import coredata
from . import mesonlib
from . import mparser
from . import mlog
from .interpreterbase import FeatureNew, FeatureDeprecated, typed_pos_args, typed_kwargs, ContainerTypeInfo, KwargInfo
from .interpreter.type_checking import NoneType, in_set_validator
@typed_kwargs('combo option', KwargInfo('value', (str, NoneType)), KwargInfo('choices', ContainerTypeInfo(list, str, allow_empty=False), required=True))
def combo_parser(self, description: str, args: T.Tuple[bool, _DEPRECATED_ARGS], kwargs: ComboArgs) -> coredata.UserOption:
    choices = kwargs['choices']
    value = kwargs['value']
    if value is None:
        value = kwargs['choices'][0]
    return coredata.UserComboOption(description, choices, value, *args)