import re
from numba import njit
from numba.core.extending import overload
from numba.core.targetconfig import ConfigStack
from numba.core.compiler import Flags, DEFAULT_FLAGS
from numba.core import types
from numba.core.funcdesc import default_mangler
from numba.tests.support import TestCase, unittest
def test_demangle(self):

    def check(flags):
        mangled = flags.get_mangle_string()
        out = flags.demangle(mangled)
        self.assertEqual(out, flags.summary())
    flags = Flags()
    check(flags)
    check(DEFAULT_FLAGS)
    flags = Flags()
    flags.no_cpython_wrapper = True
    flags.nrt = True
    flags.fastmath = True
    check(flags)