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
def test_isotonic_regression_ties_secondary_():
    """
    Test isotonic regression fit, transform  and fit_transform
    against the "secondary" ties method and "pituitary" data from R
     "isotone" package, as detailed in: J. d. Leeuw, K. Hornik, P. Mair,
     Isotone Optimization in R: Pool-Adjacent-Violators Algorithm
    (PAVA) and Active Set Methods

    Set values based on pituitary example and
     the following R command detailed in the paper above:
    > library("isotone")
    > data("pituitary")
    > res1 <- gpava(pituitary$age, pituitary$size, ties="secondary")
    > res1$x

    `isotone` version: 1.0-2, 2014-09-07
    R version: R version 3.1.1 (2014-07-10)
    """
    x = [8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14]
    y = [21, 23.5, 23, 24, 21, 25, 21.5, 22, 19, 23.5, 25]
    y_true = [22.22222, 22.22222, 22.22222, 22.22222, 22.22222, 22.22222, 22.22222, 22.22222, 22.22222, 24.25, 24.25]
    ir = IsotonicRegression()
    ir.fit(x, y)
    assert_array_almost_equal(ir.transform(x), y_true, 4)
    assert_array_almost_equal(ir.fit_transform(x, y), y_true, 4)