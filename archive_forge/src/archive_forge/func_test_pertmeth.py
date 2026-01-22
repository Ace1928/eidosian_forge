import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
def test_pertmeth(self):
    df = gendat()
    orig = df.copy()
    mx = pd.notnull(df)
    nrow, ncol = df.shape
    for pert_meth in ('gaussian', 'boot'):
        imp_data = mice.MICEData(df, perturbation_method=pert_meth)
        for k in range(2):
            imp_data.update_all()
            assert_equal(imp_data.data.shape[0], nrow)
            assert_equal(imp_data.data.shape[1], ncol)
            assert_allclose(orig[mx], imp_data.data[mx])
    assert tuple(imp_data._cycle_order) in (('x5', 'x3', 'x4', 'y', 'x2', 'x1'), ('x5', 'x4', 'x3', 'y', 'x2', 'x1'))