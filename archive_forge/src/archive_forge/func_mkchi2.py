import warnings
import numpy as np
import pytest
import scipy.stats
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection._univariate_selection import _chisquare
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
def mkchi2(k):
    """Make k-best chi2 selector"""
    return SelectKBest(chi2, k=k)