import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
@pytest.mark.parametrize('allow_none', (True, False))
@pytest.mark.parametrize('allow_auto', (True, False))
@pytest.mark.parametrize('value', [(1, 2), 'auto', None, '(1, 4)'])
def test_make_iterable_validator_none_auto(value, allow_auto, allow_none):
    scalar_validator = _validate_float_or_none
    validate_iterable = make_iterable_validator(scalar_validator, allow_auto=allow_auto, allow_none=allow_none)
    raise_error = False
    if value is None and (not allow_none):
        raise_error = 'Only ordered iterable'
    if value == 'auto' and (not allow_auto):
        raise_error = 'Could not convert'
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            validate_iterable(value)
    else:
        value = validate_iterable(value)
        assert np.iterable(value) or value is None or value == 'auto'