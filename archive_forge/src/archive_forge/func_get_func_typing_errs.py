import os, sys, subprocess
import dis
import itertools
import numpy as np
import numba
from numba import jit, njit
from numba.core import errors, ir, types, typing, typeinfer, utils
from numba.core.typeconv import Conversion
from numba.extending import overload_method
from numba.tests.support import TestCase, tag
from numba.tests.test_typeconv import CompatibilityTestMixin
from numba.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.core.typed_passes import PartialTypeInference
from numba.core.compiler_machinery import FunctionPass, register_pass
import unittest
def get_func_typing_errs(func, arg_types):
    """
    Get typing errors for function 'func'. It creates a pipeline that runs untyped
    passes as well as type inference.
    """
    typingctx = numba.core.registry.cpu_target.typing_context
    targetctx = numba.core.registry.cpu_target.target_context
    library = None
    return_type = None
    _locals = {}
    flags = numba.core.compiler.Flags()
    flags.nrt = True
    pipeline = TyperCompiler(typingctx, targetctx, library, arg_types, return_type, flags, _locals)
    pipeline.compile_extra(func)
    return pipeline.state.typing_errors