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
def insert_const_bytes(self, mod, bytes, name=None):
    """
        Insert constant *byte* (a `bytes` object) into module *mod*.
        """
    stringtype = GENERIC_POINTER
    name = '.bytes.%s' % (name or hash(bytes))
    text = cgutils.make_bytearray(bytes)
    gv = self.insert_unique_const(mod, name, text)
    return Constant.bitcast(gv, stringtype)