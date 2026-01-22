from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
@pytest.mark.parametrize(['dim', 'expected'], [pytest.param('a', ('a',), id='str'), pytest.param(['a', 'b'], ('a', 'b'), id='list'), pytest.param([...], ('a', 'b', 'c'), id='list_only_ellipsis'), pytest.param(['a', ...], ('a', 'b', 'c'), id='list_with_ellipsis'), pytest.param(['a', ..., 'b'], ('a', 'c', 'b'), id='list_with_middle_ellipsis')])
def test_parse_ordered_dims(dim, expected) -> None:
    all_dims = ('a', 'b', 'c')
    actual = utils.parse_ordered_dims(dim, all_dims)
    assert actual == expected