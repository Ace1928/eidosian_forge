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
def get_args_kwargs(sig):
    kws = []
    args = []
    pos_arg = None
    for x in sig.parameters.values():
        if x.default == utils.pyParameter.empty:
            args.append(x)
            if x.kind == utils.pyParameter.VAR_POSITIONAL:
                pos_arg = x
            elif x.kind == utils.pyParameter.VAR_KEYWORD:
                msg = "The use of VAR_KEYWORD (e.g. **kwargs) is unsupported. (offending argument name is '%s')"
                raise InternalError(msg % x)
        else:
            kws.append(x)
    return (args, kws, pos_arg)