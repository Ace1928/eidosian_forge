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
@lower_builtin((self.key, attr), self.key, types.VarArg(types.Any))
def method_impl(context, builder, sig, args):
    typ = sig.args[0]
    typing_context = context.typing_context
    fnty = self._get_function_type(typing_context, typ)
    sig = self._get_signature(typing_context, fnty, sig.args, {})
    call = context.get_function(fnty, sig)
    context.add_linking_libs(getattr(call, 'libs', ()))
    return call(builder, args)