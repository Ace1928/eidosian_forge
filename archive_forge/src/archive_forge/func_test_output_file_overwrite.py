import tempfile
import shutil
import os
import numpy as np
from numpy import pi
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.odr import (Data, Model, ODR, RealData, OdrStop, OdrWarning,
@pytest.mark.skipif(True, reason='Fortran I/O prone to crashing so better not to run this test, see gh-13127')
def test_output_file_overwrite(self):
    """
        Verify fix for gh-1892
        """

    def func(b, x):
        return b[0] + b[1] * x
    p = Model(func)
    data = Data(np.arange(10), 12 * np.arange(10))
    tmp_dir = tempfile.mkdtemp()
    error_file_path = os.path.join(tmp_dir, 'error.dat')
    report_file_path = os.path.join(tmp_dir, 'report.dat')
    try:
        ODR(data, p, beta0=[0.1, 13], errfile=error_file_path, rptfile=report_file_path).run()
        ODR(data, p, beta0=[0.1, 13], errfile=error_file_path, rptfile=report_file_path, overwrite=True).run()
    finally:
        shutil.rmtree(tmp_dir)