import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
@pytest.mark.parametrize('allow_none', (True, False))
@pytest.mark.parametrize('args', [(False, None), (False, 'row'), (False, '54row'), (False, '4column'), ('or in regex', 'square')])
def test_make_validate_choice_regex(args, allow_none):
    accepted_values = {'row', 'column'}
    accepted_values_regex = {'\\d*row', '\\d*column'}
    validate_choice = _make_validate_choice_regex(accepted_values, accepted_values_regex, allow_none=allow_none)
    raise_error, value = args
    if value is None and (not allow_none):
        raise_error = 'or in regex'
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            validate_choice(value)
    else:
        value_result = validate_choice(value)
        assert value == value_result