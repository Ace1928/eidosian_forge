from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
def check_smw_logdet(p, q, r, s):
    d = q - r
    A = np.random.normal(size=(p, q))
    AtA = np.dot(A.T, A)
    B = np.zeros((q, q))
    c = np.random.normal(size=(r, r))
    B[0:r, 0:r] = np.dot(c.T, c)
    di = np.random.uniform(size=d)
    B[r:q, r:q] = np.diag(1 / di)
    Qi = np.linalg.inv(B[0:r, 0:r])
    s = 0.5
    _, d2 = np.linalg.slogdet(s * np.eye(p, p) + np.dot(A, np.dot(B, A.T)))
    _, bd = np.linalg.slogdet(B)
    d1 = _smw_logdet(s, A, AtA, Qi, di, bd)
    rtol = 1e-06 if PLATFORM_OSX else 1e-07
    assert_allclose(d1, d2, rtol=rtol)