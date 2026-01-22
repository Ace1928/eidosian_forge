import re
import numpy as np
from numba.tests.support import (TestCase, override_config, captured_stdout,
from numba import jit, njit
from numba.core import types, ir, postproc, compiler
from numba.core.ir_utils import (guard, find_callname, find_const,
from numba.core.registry import CPUDispatcher
from numba.core.inline_closurecall import inline_closure_call
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
import unittest
def test_multiple_inner_functions(self):
    from numba.tests.inlining_usecases import outer_multiple, __name__ as prefix
    with override_config('DUMP_ASSEMBLY', True):
        with captured_stdout() as out:
            cfunc = jit((types.int32,), nopython=True)(outer_multiple)
    self.assertPreciseEqual(cfunc(1), 6)
    asm = out.getvalue()
    self.assert_has_pattern('%s.outer_multiple' % prefix, asm)
    self.assert_not_has_pattern('%s.more' % prefix, asm)
    self.assert_not_has_pattern('%s.inner' % prefix, asm)