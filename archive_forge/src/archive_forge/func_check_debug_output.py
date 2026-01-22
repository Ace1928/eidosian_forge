import numpy as np
from numba.core.utils import PYVERSION
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.tests.support import (override_config, captured_stderr,
from numba import cuda, float64
import unittest
def check_debug_output(self, out, enabled_dumps):
    all_dumps = dict.fromkeys(['bytecode', 'cfg', 'ir', 'llvm', 'assembly'], False)
    for name in enabled_dumps:
        assert name in all_dumps
        all_dumps[name] = True
    for name, enabled in sorted(all_dumps.items()):
        check_meth = getattr(self, '_check_dump_%s' % name)
        if enabled:
            check_meth(out)
        else:
            self.assertRaises(AssertionError, check_meth, out)