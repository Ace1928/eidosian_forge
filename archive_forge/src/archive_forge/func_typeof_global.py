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
def typeof_global(self, inst, target, gvar):
    try:
        typ = self.resolve_value_type(inst, gvar.value)
    except TypingError as e:
        if gvar.name == self.func_id.func_name and gvar.name in _temporary_dispatcher_map:
            typ = types.Dispatcher(_temporary_dispatcher_map[gvar.name])
        else:
            from numba.misc import special
            nm = gvar.name
            func_glbls = self.func_id.func.__globals__
            if nm not in func_glbls.keys() and nm not in special.__all__ and (nm not in __builtins__.keys()) and (nm not in self.func_id.code.co_freevars):
                errstr = "NameError: name '%s' is not defined"
                msg = _termcolor.errmsg(errstr % nm)
                e.patch_message(msg)
                raise
            else:
                msg = _termcolor.errmsg("Untyped global name '%s':" % nm)
            msg += ' %s'
            if nm in special.__all__:
                tmp = "\n'%s' looks like a Numba internal function, has it been imported (i.e. 'from numba import %s')?\n" % (nm, nm)
                msg += _termcolor.errmsg(tmp)
            e.patch_message(msg % e)
            raise
    if isinstance(typ, types.Dispatcher) and typ.dispatcher.is_compiling:
        callstack = self.context.callstack
        callframe = callstack.findfirst(typ.dispatcher.py_func)
        if callframe is not None:
            typ = types.RecursiveCall(typ)
        else:
            raise NotImplementedError('call to %s: unsupported recursion' % typ.dispatcher)
    if isinstance(typ, types.Array):
        typ = typ.copy(readonly=True)
    if isinstance(typ, types.BaseAnonymousTuple):
        literaled = [types.maybe_literal(x) for x in gvar.value]
        if all(literaled):
            typ = types.Tuple(literaled)

        def mark_array_ro(tup):
            newtup = []
            for item in tup.types:
                if isinstance(item, types.Array):
                    item = item.copy(readonly=True)
                elif isinstance(item, types.BaseAnonymousTuple):
                    item = mark_array_ro(item)
                newtup.append(item)
            return types.BaseTuple.from_types(newtup)
        typ = mark_array_ro(typ)
    self.sentry_modified_builtin(inst, gvar)
    lit = types.maybe_literal(gvar.value)
    tv = self.typevars[target.name]
    if tv.locked:
        tv.add_type(lit or typ, loc=inst.loc)
    else:
        self.lock_type(target.name, lit or typ, loc=inst.loc)
    self.assumed_immutables.add(inst)