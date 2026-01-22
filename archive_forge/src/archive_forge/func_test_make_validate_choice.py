import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
@pytest.mark.parametrize('allow_none', (True, False))
@pytest.mark.parametrize('typeof', (str, int))
@pytest.mark.parametrize('args', [('not one', 10), (False, None), (False, 4)])
def test_make_validate_choice(args, allow_none, typeof):
    accepted_values = set((typeof(value) for value in (0, 1, 4, 6)))
    validate_choice = _make_validate_choice(accepted_values, allow_none=allow_none, typeof=typeof)
    raise_error, value = args
    if value is None and (not allow_none):
        raise_error = 'not one of' if typeof == str else 'Could not convert'
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            validate_choice(value)
    else:
        value = validate_choice(value)
        assert value in accepted_values or value is None