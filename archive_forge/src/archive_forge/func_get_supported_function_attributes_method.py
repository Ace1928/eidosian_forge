from __future__ import annotations
import collections
import enum
import functools
import os
import itertools
import typing as T
from .. import build
from .. import coredata
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..compilers import SUFFIX_TO_LANG
from ..compilers.compilers import CompileCheckMode
from ..interpreterbase import (ObjectHolder, noPosargs, noKwargs,
from ..interpreterbase.decorators import ContainerTypeInfo, typed_kwargs, KwargInfo, typed_pos_args
from ..mesonlib import OptionKey
from .interpreterobjects import (extract_required_kwarg, extract_search_dirs)
from .type_checking import REQUIRED_KW, in_set_validator, NoneType
@FeatureNew('compiler.get_supported_function_attributes', '0.48.0')
@noKwargs
@typed_pos_args('compiler.get_supported_function_attributes', varargs=str)
def get_supported_function_attributes_method(self, args: T.Tuple[T.List[str]], kwargs: 'TYPE_kwargs') -> T.List[str]:
    return [a for a in args[0] if self._has_function_attribute_impl(a)]