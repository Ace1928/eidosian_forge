from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
def test_parse_ordered_dims_raises() -> None:
    all_dims = ('a', 'b', 'c')
    with pytest.raises(ValueError, match="'x' do not exist"):
        utils.parse_ordered_dims('x', all_dims, check_exists=True)
    with pytest.raises(ValueError, match='repeated dims'):
        utils.parse_ordered_dims(['a', ...], all_dims + ('a',))
    with pytest.raises(ValueError, match='More than one ellipsis'):
        utils.parse_ordered_dims(['a', ..., 'b', ...], all_dims)