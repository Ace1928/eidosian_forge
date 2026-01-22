import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
def get_generator_type(self, typdict, retty, raise_errors=True):
    gi = self.generator_info
    arg_types = [None] * len(self.arg_names)
    for index, name in self.arg_names.items():
        arg_types[index] = typdict[name]
    state_types = None
    try:
        state_types = [typdict[var_name] for var_name in gi.state_vars]
    except KeyError:
        msg = 'Cannot type generator: state variable types cannot be found'
        if raise_errors:
            raise TypingError(msg)
        state_types = [types.unknown for _ in gi.state_vars]
    yield_types = None
    try:
        yield_types = [typdict[y.inst.value.name] for y in gi.get_yield_points()]
    except KeyError:
        msg = 'Cannot type generator: yield type cannot be found'
        if raise_errors:
            raise TypingError(msg)
    if not yield_types:
        msg = 'Cannot type generator: it does not yield any value'
        if raise_errors:
            raise TypingError(msg)
        yield_types = [types.unknown for _ in gi.get_yield_points()]
    if not yield_types or all(yield_types) == types.unknown:
        return types.Generator(self.func_id.func, types.unknown, arg_types, state_types, has_finalizer=True)
    yield_type = self.context.unify_types(*yield_types)
    if yield_type is None or isinstance(yield_type, types.Optional):
        msg = 'Cannot type generator: cannot unify yielded types %s'
        yp_highlights = []
        for y in gi.get_yield_points():
            msg = _termcolor.errmsg("Yield of: IR '%s', type '%s', location: %s")
            yp_highlights.append(msg % (str(y.inst), typdict[y.inst.value.name], y.inst.loc.strformat()))
        explain_ty = set()
        for ty in yield_types:
            if isinstance(ty, types.Optional):
                explain_ty.add(ty.type)
                explain_ty.add(types.NoneType('none'))
            else:
                explain_ty.add(ty)
        if raise_errors:
            raise TypingError("Can't unify yield type from the following types: %s" % ', '.join(sorted(map(str, explain_ty))) + '\n\n' + '\n'.join(yp_highlights))
    return types.Generator(self.func_id.func, yield_type, arg_types, state_types, has_finalizer=True)