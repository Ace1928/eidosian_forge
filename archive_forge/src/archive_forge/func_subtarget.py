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
def subtarget(self, **kws):
    obj = copy.copy(self)
    for k, v in kws.items():
        if not hasattr(obj, k):
            raise NameError('unknown option {0!r}'.format(k))
        setattr(obj, k, v)
    if obj.codegen() is not self.codegen():
        obj.cached_internal_func = {}
    return obj