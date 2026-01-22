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
def call_internal(self, builder, fndesc, sig, args):
    """
        Given the function descriptor of an internally compiled function,
        emit a call to that function with the given arguments.
        """
    status, res = self.call_internal_no_propagate(builder, fndesc, sig, args)
    with cgutils.if_unlikely(builder, status.is_error):
        self.call_conv.return_status_propagate(builder, status)
    res = imputils.fix_returning_optional(self, builder, sig, status, res)
    return res