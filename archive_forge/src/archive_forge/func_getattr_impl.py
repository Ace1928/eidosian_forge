from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
@lower_getattr(cls.key, attr)
def getattr_impl(context, builder, typ, value):
    typingctx = context.typing_context
    fnty = cls._get_function_type(typingctx, typ)
    sig = cls._get_signature(typingctx, fnty, (typ,), {})
    call = context.get_function(fnty, sig)
    return call(builder, (value,))