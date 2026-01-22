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
@FeatureNew('compiler.has_function_attribute', '0.48.0')
@typed_pos_args('compiler.has_function_attribute', str)
@typed_kwargs('compiler.has_function_attribute', _HAS_REQUIRED_KW)
def has_func_attribute_method(self, args: T.Tuple[str], kwargs: 'HasArgumentKW') -> bool:
    return self._has_function_attribute_impl(args[0], kwargs)