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
def declare_env_global(self, module, envname):
    """Declare the Environment pointer as a global of the module.

        The pointer is initialized to NULL.  It must be filled by the runtime
        with the actual address of the Env before the associated function
        can be executed.

        Parameters
        ----------
        module :
            The LLVM Module
        envname : str
            The name of the global variable.
        """
    if envname not in module.globals:
        gv = llvmir.GlobalVariable(module, cgutils.voidptr_t, name=envname)
        gv.linkage = 'common'
        gv.initializer = cgutils.get_null_value(gv.type.pointee)
    return module.globals[envname]