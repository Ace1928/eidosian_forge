import numpy as np
import pytest
from sklearn.impute._base import _BaseImputer
from sklearn.impute._iterative import _assign_where
from sklearn.utils._mask import _get_mask
from sklearn.utils._testing import _convert_container, assert_allclose
def test_base_no_precomputed_mask_transform(data):
    imputer = NoPrecomputedMaskTransform(add_indicator=True)
    err_msg = 'precomputed is True but the input data is not a mask'
    imputer.fit(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)