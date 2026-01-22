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
def sentry_record_alignment(self, rectyp, attr):
    """
        Assumes offset starts from a properly aligned location
        """
    if self.strict_alignment:
        offset = rectyp.offset(attr)
        elemty = rectyp.typeof(attr)
        if isinstance(elemty, types.NestedArray):
            elemty = elemty.dtype
        align = self.get_abi_alignment(self.get_data_type(elemty))
        if offset % align:
            msg = '{rec}.{attr} of type {type} is not aligned'.format(rec=rectyp, attr=attr, type=elemty)
            raise TypeError(msg)