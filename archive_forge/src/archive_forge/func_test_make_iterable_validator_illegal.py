import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
@pytest.mark.parametrize('args', [('Only ordered iterable', set(['a', 'b', 'c'])), ('Could not convert', 'johndoe'), ('Only ordered iterable', 15)])
def test_make_iterable_validator_illegal(args):
    scalar_validator = _validate_float_or_none
    validate_iterable = make_iterable_validator(scalar_validator)
    raise_error, value = args
    with pytest.raises(ValueError, match=raise_error):
        validate_iterable(value)