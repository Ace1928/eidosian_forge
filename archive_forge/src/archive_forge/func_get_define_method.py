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
@FeatureNew('compiler.get_define', '0.40.0')
@typed_pos_args('compiler.get_define', str)
@typed_kwargs('compiler.get_define', *_COMMON_KWS)
def get_define_method(self, args: T.Tuple[str], kwargs: 'CommonKW') -> str:
    element = args[0]
    extra_args = functools.partial(self._determine_args, kwargs)
    deps, msg = self._determine_dependencies(kwargs['dependencies'])
    value, cached = self.compiler.get_define(element, kwargs['prefix'], self.environment, extra_args=extra_args, dependencies=deps)
    cached_msg = mlog.blue('(cached)') if cached else ''
    value_msg = '(undefined)' if value is None else value
    mlog.log('Fetching value of define', mlog.bold(element, True), msg, value_msg, cached_msg)
    return value if value is not None else ''