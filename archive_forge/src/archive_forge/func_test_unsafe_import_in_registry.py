import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_unsafe_import_in_registry(self):
    code = dedent('\n            import numba\n            import numpy as np\n            @numba.njit\n            def foo():\n                np.array([1 for _ in range(1)])\n            foo()\n            print("OK")\n        ')
    result, error = run_in_subprocess(code)
    self.assertEqual(b'OK', result.strip())
    self.assertEqual(b'', error.strip())