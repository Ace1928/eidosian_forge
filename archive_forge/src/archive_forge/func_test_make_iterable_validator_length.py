import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
@pytest.mark.parametrize('length', (2, None))
@pytest.mark.parametrize('value', [(1, 5), (1, 3, 5), '(3, 4, 5)'])
def test_make_iterable_validator_length(value, length):
    scalar_validator = _validate_float_or_none
    validate_iterable = make_iterable_validator(scalar_validator, length=length)
    raise_error = False
    if length is not None and len(value) != length:
        raise_error = 'Iterable must be of length'
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            validate_iterable(value)
    else:
        value = validate_iterable(value)
        assert np.iterable(value)