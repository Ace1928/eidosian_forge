import os
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pandas as pd
import pytest
import patsy
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen
from statsmodels.gam.smooth_basis import (BSplines, CyclicCubicSplines)
from statsmodels.gam.generalized_additive_model import (
from statsmodels.tools.linalg import matrix_sqrt, transf_constraints
from .results import results_pls, results_mpg_bs, results_mpg_bs_poisson
def test_crossval(self):
    mod = self.res1.model
    assert_equal(mod.alpha, self.alpha)
    assert_allclose(self.res1.scale, 4.706482135439112, rtol=1e-13)
    alpha_aic = mod.select_penweight()[0]
    assert_allclose(alpha_aic, [112487.81362014, 129.89155677], rtol=0.001)
    assert_equal(mod.alpha, self.alpha)
    assert_equal(mod.penal.start_idx, 4)
    pm = mod.penal.penalty_matrix()
    assert_equal(pm[:, :4], 0)
    assert_equal(pm[:4, :], 0)
    assert_allclose(self.res1.scale, 4.706482135439112, rtol=1e-13)
    np.random.seed(987125)
    alpha_cv, _ = mod.select_penweight_kfold(k_folds=3, k_grid=6)
    assert_allclose(alpha_cv, [10000000.0, 630.957344480193], rtol=1e-05)
    assert_equal(mod.alpha, self.alpha)
    assert_equal(mod.penal.start_idx, 4)
    pm = mod.penal.penalty_matrix()
    assert_equal(pm[:, :4], 0)
    assert_equal(pm[:4, :], 0)
    assert_allclose(self.res1.scale, 4.706482135439112, rtol=1e-13)