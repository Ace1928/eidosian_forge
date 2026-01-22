from statsmodels.compat.platform import PLATFORM_WIN32
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.multivariate.pca import PCA, pca
from statsmodels.multivariate.tests.results.datamlw import (data, princomp1,
from statsmodels.tools.sm_exceptions import EstimationWarning
@pytest.mark.slow
def test_missing_dataframe(self):
    x = self.x.copy()
    x[::5, ::7] = np.nan
    pc = PCA(x, ncomp=3, missing='fill-em')
    x = pd.DataFrame(x)
    pc_df = PCA(x, ncomp=3, missing='fill-em')
    assert_allclose(pc.coeff, pc_df.coeff)
    assert_allclose(pc.factors, pc_df.factors)
    pc_df_nomissing = PCA(pd.DataFrame(self.x.copy()), ncomp=3)
    assert isinstance(pc_df.coeff, type(pc_df_nomissing.coeff))
    assert isinstance(pc_df.data, type(pc_df_nomissing.data))
    assert isinstance(pc_df.eigenvals, type(pc_df_nomissing.eigenvals))
    assert isinstance(pc_df.eigenvecs, type(pc_df_nomissing.eigenvecs))
    x = self.x.copy()
    x[::5, ::7] = np.nan
    x_df = pd.DataFrame(x)
    pc = PCA(x, missing='drop-row')
    pc_df = PCA(x_df, missing='drop-row')
    assert_allclose(pc.coeff, pc_df.coeff)
    assert_allclose(pc.factors, pc_df.factors)
    pc = PCA(x, missing='drop-col')
    pc_df = PCA(x_df, missing='drop-col')
    assert_allclose(pc.coeff, pc_df.coeff)
    assert_allclose(pc.factors, pc_df.factors)
    pc = PCA(x, missing='drop-min')
    pc_df = PCA(x_df, missing='drop-min')
    assert_allclose(pc.coeff, pc_df.coeff)
    assert_allclose(pc.factors, pc_df.factors)