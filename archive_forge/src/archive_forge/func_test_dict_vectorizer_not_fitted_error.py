from random import Random
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
@pytest.mark.parametrize('method, input', [('transform', [{1: 2, 3: 4}, {2: 4}]), ('inverse_transform', [{1: 2, 3: 4}, {2: 4}]), ('restrict', [True, False, True])])
def test_dict_vectorizer_not_fitted_error(method, input):
    """Check that unfitted DictVectorizer instance raises NotFittedError.

    This should be part of the common test but currently they test estimator accepting
    text input.
    """
    dv = DictVectorizer(sparse=False)
    with pytest.raises(NotFittedError):
        getattr(dv, method)(input)