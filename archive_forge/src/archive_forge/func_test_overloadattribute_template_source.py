import sys
import subprocess
import numpy as np
import os
import warnings
from numba import jit, njit, types
from numba.core import errors
from numba.experimental import structref
from numba.extending import (overload, intrinsic, overload_method,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, DeadCodeElimination,
from numba.core.compiler_machinery import PassManager
from numba.core.types.functions import _err_reasons as error_reasons
from numba.tests.support import (skip_parfors_unsupported, override_config,
import unittest
def test_overloadattribute_template_source(self):

    @overload_attribute(types.UnicodeType, 'isnonsense')
    def ol_unicode_isnonsense(self):
        pass

    @njit
    def foo():
        'abc'.isnonsense
    with self.assertRaises(errors.TypingError) as raises:
        foo()
    excstr = str(raises.exception)
    self.assertIn("Overload of function 'ol_unicode_isnonsense'", excstr)