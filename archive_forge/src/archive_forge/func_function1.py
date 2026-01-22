import sys
import copy
import logging
import numpy as np
from numba import njit, jit, types
from numba.core import errors, ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.untyped_passes import ReconstructSSA, PreserveIR
from numba.core.typed_passes import NativeLowering
from numba.extending import overload
from numba.tests.support import MemoryLeakMixin, TestCase, override_config
def function1(arg1, arg2, arg3, arg4, arg5):
    if arg1:
        var1 = arg2
        var2 = arg3
        var3 = var2
        var4 = arg1
        return
    else:
        if arg2:
            if arg4:
                var5 = arg4
                return
            else:
                var6 = var4
                return
            return var6
        else:
            if arg5:
                if var1:
                    if arg5:
                        var1 = var6
                        return
                    else:
                        var7 = arg2
                        return arg2
                    return
                else:
                    if var2:
                        arg5 = arg2
                        return arg1
                    else:
                        var6 = var3
                        return var4
                    return
                return
            else:
                var8 = var1
                return
            return var8
        var9 = var3
        var10 = arg5
        return var1