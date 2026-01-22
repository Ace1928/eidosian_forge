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
def test_xerbla_override():
    XERBLA_OK = 255
    try:
        pid = os.fork()
    except (OSError, AttributeError):
        pytest.skip('Not POSIX or fork failed.')
    if pid == 0:
        os.close(1)
        os.close(0)
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            np.linalg.lapack_lite.xerbla()
        except ValueError:
            pass
        except Exception:
            os._exit(os.EX_CONFIG)
        try:
            a = np.array([[1.0]])
            np.linalg.lapack_lite.dorgqr(1, 1, 1, a, 0, a, a, 0, 0)
        except ValueError as e:
            if 'DORGQR parameter number 5' in str(e):
                os._exit(XERBLA_OK)
        os._exit(os.EX_CONFIG)
    else:
        pid, status = os.wait()
        if os.WEXITSTATUS(status) != XERBLA_OK:
            pytest.skip('Numpy xerbla not linked in.')