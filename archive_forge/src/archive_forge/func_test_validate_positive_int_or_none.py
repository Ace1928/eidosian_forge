import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
@pytest.mark.parametrize('args', [('Only positive', -1), ('Could not convert', '1.3'), (False, '2'), (False, None), (False, 1)])
def test_validate_positive_int_or_none(args):
    raise_error, value = args
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            _validate_positive_int_or_none(value)
    else:
        value = _validate_positive_int_or_none(value)
        assert isinstance(value, int) or value is None