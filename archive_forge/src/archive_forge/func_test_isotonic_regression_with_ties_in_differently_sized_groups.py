import copy
import pickle
import warnings
import numpy as np
import pytest
from scipy.special import expit
import sklearn
from sklearn.datasets import make_regression
from sklearn.isotonic import (
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.validation import check_array
def test_isotonic_regression_with_ties_in_differently_sized_groups():
    """
    Non-regression test to handle issue 9432:
    https://github.com/scikit-learn/scikit-learn/issues/9432

    Compare against output in R:
    > library("isotone")
    > x <- c(0, 1, 1, 2, 3, 4)
    > y <- c(0, 0, 1, 0, 0, 1)
    > res1 <- gpava(x, y, ties="secondary")
    > res1$x

    `isotone` version: 1.1-0, 2015-07-24
    R version: R version 3.3.2 (2016-10-31)
    """
    x = np.array([0, 1, 1, 2, 3, 4])
    y = np.array([0, 0, 1, 0, 0, 1])
    y_true = np.array([0.0, 0.25, 0.25, 0.25, 0.25, 1.0])
    ir = IsotonicRegression()
    ir.fit(x, y)
    assert_array_almost_equal(ir.transform(x), y_true)
    assert_array_almost_equal(ir.fit_transform(x, y), y_true)