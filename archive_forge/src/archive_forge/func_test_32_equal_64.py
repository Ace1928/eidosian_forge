import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.parametrize('input_dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('encode', ['ordinal', 'onehot', 'onehot-dense'])
def test_32_equal_64(input_dtype, encode):
    X_input = np.array(X, dtype=input_dtype)
    kbd_32 = KBinsDiscretizer(n_bins=3, encode=encode, dtype=np.float32)
    kbd_32.fit(X_input)
    Xt_32 = kbd_32.transform(X_input)
    kbd_64 = KBinsDiscretizer(n_bins=3, encode=encode, dtype=np.float64)
    kbd_64.fit(X_input)
    Xt_64 = kbd_64.transform(X_input)
    assert_allclose_dense_sparse(Xt_32, Xt_64)