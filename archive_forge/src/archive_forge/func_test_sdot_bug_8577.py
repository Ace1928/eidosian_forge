import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
@pytest.mark.skipif(IS_WASM, reason='Cannot start subprocess')
@pytest.mark.slow
def test_sdot_bug_8577():
    bad_libs = ['PyQt5.QtWidgets', 'IPython']
    template = textwrap.dedent('\n    import sys\n    {before}\n    try:\n        import {bad_lib}\n    except ImportError:\n        sys.exit(0)\n    {after}\n    x = np.ones(2, dtype=np.float32)\n    sys.exit(0 if np.allclose(x.dot(x), 2.0) else 1)\n    ')
    for bad_lib in bad_libs:
        code = template.format(before='import numpy as np', after='', bad_lib=bad_lib)
        subprocess.check_call([sys.executable, '-c', code])
        code = template.format(after='import numpy as np', before='', bad_lib=bad_lib)
        subprocess.check_call([sys.executable, '-c', code])