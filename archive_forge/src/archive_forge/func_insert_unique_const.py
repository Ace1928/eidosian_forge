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
def insert_unique_const(self, mod, name, val):
    """
        Insert a unique internal constant named *name*, with LLVM value
        *val*, into module *mod*.
        """
    try:
        gv = mod.get_global(name)
    except KeyError:
        return cgutils.global_constant(mod, name, val)
    else:
        return gv