import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
def test_invalid_kernel():
    x = np.arange(400)
    y = x ** 2
    with pytest.raises(ValueError):
        nparam.KernelReg(x, y, reg_type='ll', var_type='cc', bw='cv_ls', ckertype='silverman')
    with pytest.raises(ValueError):
        nparam.KernelCensoredReg(x, y, reg_type='ll', var_type='cc', bw='cv_ls', censor_val=0, ckertype='silverman')