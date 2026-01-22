import json
import os
import warnings
import tempfile
from functools import wraps
import numpy as np
import numpy.array_api
import numpy.testing as npt
import pytest
import hypothesis
from scipy._lib._fpumode import get_fpu_mode
from scipy._lib._testutils import FPUModeChangeWarning
from scipy._lib import _pep440
from scipy._lib._array_api import SCIPY_ARRAY_API, SCIPY_DEVICE
def pytest_runtest_setup(item):
    mark = _get_mark(item, 'xslow')
    if mark is not None:
        try:
            v = int(os.environ.get('SCIPY_XSLOW', '0'))
        except ValueError:
            v = False
        if not v:
            pytest.skip('very slow test; set environment variable SCIPY_XSLOW=1 to run it')
    mark = _get_mark(item, 'xfail_on_32bit')
    if mark is not None and np.intp(0).itemsize < 8:
        pytest.xfail(f'Fails on our 32-bit test platform(s): {mark.args[0]}')
    with npt.suppress_warnings() as sup:
        sup.filter(pytest.PytestUnraisableExceptionWarning)
        try:
            from threadpoolctl import threadpool_limits
            HAS_THREADPOOLCTL = True
        except Exception:
            HAS_THREADPOOLCTL = False
        if HAS_THREADPOOLCTL:
            try:
                xdist_worker_count = int(os.environ['PYTEST_XDIST_WORKER_COUNT'])
            except KeyError:
                return
            if not os.getenv('OMP_NUM_THREADS'):
                max_openmp_threads = os.cpu_count() // 2
                threads_per_worker = max(max_openmp_threads // xdist_worker_count, 1)
                try:
                    threadpool_limits(threads_per_worker, user_api='blas')
                except Exception:
                    return