from statsmodels.compat.pandas import MONTH_END
import warnings
import numpy as np
from numpy.testing import assert_, assert_allclose
import pandas as pd
import pytest
from scipy.stats import ortho_group
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tsa.statespace import (
from statsmodels.tsa.vector_ar.tests.test_var import get_macrodata
def test_time_varying_out_of_sample_anchored(reset_randomstate):
    mod = TVSS(np.zeros((10, 2)))
    anchor = 2
    new_Z = mod['design', :, :, -1]
    new_T = mod['transition', :, :, -1]
    irfs = mod.impulse_responses([], steps=mod.nobs - anchor, anchor=anchor, design=new_Z[:, :, None], transition=new_T[:, :, None])
    cirfs = mod.impulse_responses([], steps=mod.nobs - anchor, anchor=anchor, design=new_Z[:, :, None], transition=new_T[:, :, None], cumulative=True)
    oirfs = mod.impulse_responses([], steps=mod.nobs - anchor, anchor=anchor, design=new_Z[:, :, None], transition=new_T[:, :, None], orthogonalized=True)
    coirfs = mod.impulse_responses([], steps=mod.nobs - anchor, anchor=anchor, design=new_Z[:, :, None], transition=new_T[:, :, None], cumulative=True, orthogonalized=True)
    Z = mod['design']
    T = mod['transition']
    R = mod['selection']
    Q = mod['state_cov', ..., anchor]
    L = np.linalg.cholesky(Q)
    desired_irfs = np.zeros((mod.nobs - anchor, 2)) * np.nan
    desired_oirfs = np.zeros((mod.nobs - anchor, 2)) * np.nan
    tmp = R[..., anchor]
    for i in range(1, mod.nobs - anchor):
        desired_irfs[i - 1] = Z[:, :, i + anchor].dot(tmp)[:, 0]
        desired_oirfs[i - 1] = Z[:, :, i + anchor].dot(tmp).dot(L)[:, 0]
        tmp = T[:, :, i + anchor].dot(tmp)
    desired_irfs[mod.nobs - anchor - 1] = new_Z.dot(tmp)[:, 0]
    desired_oirfs[mod.nobs - anchor - 1] = new_Z.dot(tmp).dot(L)[:, 0]
    assert_allclose(irfs, desired_irfs)
    assert_allclose(cirfs, np.cumsum(desired_irfs, axis=0))
    assert_allclose(oirfs, desired_oirfs)
    assert_allclose(coirfs, np.cumsum(desired_oirfs, axis=0))