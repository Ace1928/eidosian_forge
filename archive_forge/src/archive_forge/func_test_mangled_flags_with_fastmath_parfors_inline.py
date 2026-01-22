import re
from numba import njit
from numba.core.extending import overload
from numba.core.targetconfig import ConfigStack
from numba.core.compiler import Flags, DEFAULT_FLAGS
from numba.core import types
from numba.core.funcdesc import default_mangler
from numba.tests.support import TestCase, unittest
def test_mangled_flags_with_fastmath_parfors_inline(self):
    flags = Flags()
    flags.nrt = True
    flags.auto_parallel = True
    flags.fastmath = True
    flags.inline = 'always'
    self.assertLess(len(flags.get_mangle_string()), len(flags.summary()))
    demangled = flags.demangle(flags.get_mangle_string())
    self.assertNotIn('0x', demangled)