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
def fold_arg_vars(typevars, args, vararg, kws):
    """
    Fold and resolve the argument variables of a function call.
    """
    n_pos_args = len(args)
    kwds = [kw for kw, var in kws]
    argtypes = [typevars[a.name] for a in args]
    argtypes += [typevars[var.name] for kw, var in kws]
    if vararg is not None:
        argtypes.append(typevars[vararg.name])
    if not all((a.defined for a in argtypes)):
        return
    args = tuple((a.getone() for a in argtypes))
    pos_args = args[:n_pos_args]
    if vararg is not None:
        errmsg = '*args in function call should be a tuple, got %s'
        if isinstance(args[-1], types.Literal):
            const_val = args[-1].literal_value
            if not isinstance(const_val, tuple):
                raise TypeError(errmsg % (args[-1],))
            pos_args += const_val
        elif not isinstance(args[-1], types.BaseTuple):
            raise TypeError(errmsg % (args[-1],))
        else:
            pos_args += args[-1].types
        args = args[:-1]
    kw_args = dict(zip(kwds, args[n_pos_args:]))
    return (pos_args, kw_args)