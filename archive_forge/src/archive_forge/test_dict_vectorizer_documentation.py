from random import Random
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
Check that unfitted DictVectorizer instance raises NotFittedError.

    This should be part of the common test but currently they test estimator accepting
    text input.
    