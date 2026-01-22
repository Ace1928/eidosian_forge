import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('X, cat_exp, cat_dtype', [([['abc', 55], ['def', 55]], [['abc', 'def'], [55]], np.object_), (np.array([[1, 2], [3, 2]]), [[1, 3], [2]], np.integer), (np.array([['A', 'cat'], ['B', 'cat']], dtype=object), [['A', 'B'], ['cat']], np.object_), (np.array([['A', 'cat'], ['B', 'cat']]), [['A', 'B'], ['cat']], np.str_), (np.array([[1, 2], [np.nan, 2]]), [[1, np.nan], [2]], np.float64), (np.array([['A', np.nan], [None, np.nan]], dtype=object), [['A', None], [np.nan]], np.object_), (np.array([['A', float('nan')], [None, float('nan')]], dtype=object), [['A', None], [float('nan')]], np.object_)], ids=['mixed', 'numeric', 'object', 'string', 'missing-float', 'missing-np.nan-object', 'missing-float-nan-object'])
def test_one_hot_encoder_categories(X, cat_exp, cat_dtype):
    for Xi in [X, X[::-1]]:
        enc = OneHotEncoder(categories='auto')
        enc.fit(Xi)
        assert isinstance(enc.categories_, list)
        for res, exp in zip(enc.categories_, cat_exp):
            res_list = res.tolist()
            if is_scalar_nan(exp[-1]):
                assert is_scalar_nan(res_list[-1])
                assert res_list[:-1] == exp[:-1]
            else:
                assert res.tolist() == exp
            assert np.issubdtype(res.dtype, cat_dtype)