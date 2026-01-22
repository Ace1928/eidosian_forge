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
@typed_pos_args('compiler.sizeof', str)
@typed_kwargs('compiler.sizeof', *_COMMON_KWS)
def sizeof_method(self, args: T.Tuple[str], kwargs: 'CommonKW') -> int:
    element = args[0]
    extra_args = functools.partial(self._determine_args, kwargs)
    deps, msg = self._determine_dependencies(kwargs['dependencies'], compile_only=self.compiler.is_cross)
    esize, cached = self.compiler.sizeof(element, kwargs['prefix'], self.environment, extra_args=extra_args, dependencies=deps)
    cached_msg = mlog.blue('(cached)') if cached else ''
    mlog.log('Checking for size of', mlog.bold(element, True), msg, mlog.bold(str(esize)), cached_msg)
    return esize