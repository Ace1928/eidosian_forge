from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def test_numpy_endog():
    endog = np.array([1.0, 2.0])
    mod = MLEModel(endog, **kwargs)
    assert_equal(mod.endog.base is not mod.data.orig_endog, True)
    assert_equal(mod.endog.base is not endog, True)
    assert_equal(mod.data.orig_endog.base is not endog, True)
    endog[0] = 2
    assert_equal(mod.endog, np.r_[1, 2].reshape(2, 1))
    assert_equal(mod.data.orig_endog, endog)
    endog = np.array(1.0)
    assert_raises(TypeError, check_endog, endog, **kwargs)
    endog = np.array([1.0, 2.0])
    assert_equal(endog.ndim, 1)
    assert_equal(endog.flags['C_CONTIGUOUS'], True)
    assert_equal(endog.flags['F_CONTIGUOUS'], True)
    assert_equal(endog.shape, (2,))
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = np.array([1.0, 2.0]).reshape(2, 1)
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['C_CONTIGUOUS'], True)
    assert_equal(endog.shape, (2, 1))
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = np.array([1.0, 2.0]).reshape(1, 2)
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['C_CONTIGUOUS'], True)
    assert_equal(endog.shape, (1, 2))
    assert_raises(ValueError, check_endog, endog, **kwargs)
    endog = np.array([1.0, 2.0]).reshape(1, 2).transpose()
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['F_CONTIGUOUS'], True)
    assert_equal(endog.shape, (2, 1))
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = np.array([1.0, 2.0]).reshape(2, 1).transpose()
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['F_CONTIGUOUS'], True)
    assert_equal(endog.shape, (1, 2))
    assert_raises(ValueError, check_endog, endog, **kwargs)
    endog = np.array([1.0, 2.0]).reshape(2, 1, 1)
    assert_raises(ValueError, check_endog, endog, **kwargs)
    kwargs2 = {'k_states': 1, 'design': [[1], [0.0]], 'obs_cov': [[1, 0], [0, 1]], 'transition': [[1]], 'selection': [[1]], 'state_cov': [[1]], 'initialization': 'approximate_diffuse'}
    endog = np.array([[1.0, 2.0], [3.0, 4.0]])
    mod = check_endog(endog, k_endog=2, **kwargs2)
    mod.filter([])