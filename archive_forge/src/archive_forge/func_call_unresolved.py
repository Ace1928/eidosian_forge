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
def call_unresolved(self, builder, name, sig, args):
    """
        Insert a function call to an unresolved symbol with the given *name*.

        Note: this is used for recursive call.

        In the mutual recursion case::

            @njit
            def foo():
                ...  # calls bar()

            @njit
            def bar():
                ... # calls foo()

            foo()

        When foo() is called, the compilation of bar() is fully completed
        (codegen'ed and loaded) before foo() is. Since MCJIT's eager compilation
        doesn't allow loading modules with declare-only functions (which is
        needed for foo() in bar()), the call_unresolved injects a global
        variable that the "linker" can update even after the module is loaded by
        MCJIT. The linker would allocate space for the global variable before
        the bar() module is loaded. When later foo() module is defined, it will
        update bar()'s reference to foo().

        The legacy lazy JIT and the new ORC JIT would allow a declare-only
        function be used in a module as long as it is defined by the time of its
        first use.
        """
    codegen = self.codegen()
    fnty = self.call_conv.get_function_type(sig.return_type, sig.args)
    fn = codegen.insert_unresolved_ref(builder, fnty, name)
    status, res = self.call_conv.call_function(builder, fn, sig.return_type, sig.args, args)
    with cgutils.if_unlikely(builder, status.is_error):
        self.call_conv.return_status_propagate(builder, status)
    res = imputils.fix_returning_optional(self, builder, sig, status, res)
    return res