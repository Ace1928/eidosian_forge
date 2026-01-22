from collections import defaultdict
import copy
import sys
from itertools import permutations, takewhile
from contextlib import contextmanager
from functools import cached_property
from llvmlite import ir as llvmir
from llvmlite.ir import Constant
import llvmlite.binding as ll
from numba.core import types, utils, datamodel, debuginfo, funcdesc, config, cgutils, imputils
from numba.core import event, errors, targetconfig
from numba import _dynfunc, _helperlib
from numba.core.compiler_lock import global_compiler_lock
from numba.core.pythonapi import PythonAPI
from numba.core.imputils import (user_function, user_generator,
from numba.cpython import builtins
def get_setattr(self, attr, sig):
    """
        Get the setattr() implementation for the given attribute name
        and signature.
        The return value is a callable with the signature (builder, args).
        """
    assert len(sig.args) == 2
    typ = sig.args[0]
    valty = sig.args[1]

    def wrap_setattr(impl):

        def wrapped(builder, args):
            return impl(self, builder, sig, args, attr)
        return wrapped
    overloads = self._setattrs[attr]
    try:
        return wrap_setattr(overloads.find((typ, valty)))
    except errors.NumbaNotImplementedError:
        pass
    overloads = self._setattrs[None]
    try:
        return wrap_setattr(overloads.find((typ, valty)))
    except errors.NumbaNotImplementedError:
        pass
    raise NotImplementedError('No definition for lowering %s.%s = %s' % (typ, attr, valty))