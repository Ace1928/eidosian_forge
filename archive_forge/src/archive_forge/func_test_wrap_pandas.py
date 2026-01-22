from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def test_wrap_pandas(use_pandas):
    a = gen_data(1, use_pandas)
    b = gen_data(1, False)
    wrapped = PandasWrapper(a).wrap(b)
    expected_type = pd.Series if use_pandas else np.ndarray
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.name is None
    wrapped = PandasWrapper(a).wrap(b, columns='name')
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.name == 'name'
    wrapped = PandasWrapper(a).wrap(b, columns=['name'])
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.name == 'name'
    expected_type = pd.DataFrame if use_pandas else np.ndarray
    wrapped = PandasWrapper(a).wrap(b[:, None])
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.columns[0] == 0
    wrapped = PandasWrapper(a).wrap(b[:, None], columns=['name'])
    assert isinstance(wrapped, expected_type)
    assert not use_pandas or wrapped.columns == ['name']
    if use_pandas:
        match = 'Can only wrap 1 or 2-d array_like'
        with pytest.raises(ValueError, match=match):
            PandasWrapper(a).wrap(b[:, None, None])
        match = 'obj must have the same number of elements in axis 0 as'
        with pytest.raises(ValueError, match=match):
            PandasWrapper(a).wrap(b[:b.shape[0] // 2])