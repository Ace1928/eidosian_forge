from __future__ import annotations
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs, infix_dims, iterate_nested
from xarray.tests import assert_array_equal, requires_dask
@pytest.mark.parametrize(['dim', 'expected'], [pytest.param('a', ('a',), id='str'), pytest.param(['a', 'b'], ('a', 'b'), id='list_of_str'), pytest.param(['a', 1], ('a', 1), id='list_mixed'), pytest.param(['a', ...], ('a', ...), id='list_with_ellipsis'), pytest.param(('a', 'b'), ('a', 'b'), id='tuple_of_str'), pytest.param(['a', ('b', 'c')], ('a', ('b', 'c')), id='list_with_tuple'), pytest.param((('b', 'c'),), (('b', 'c'),), id='tuple_of_tuple'), pytest.param({'a', 1}, tuple({'a', 1}), id='non_sequence_collection'), pytest.param((), (), id='empty_tuple'), pytest.param(set(), (), id='empty_collection'), pytest.param(None, None, id='None'), pytest.param(..., ..., id='ellipsis')])
def test_parse_dims(dim, expected) -> None:
    all_dims = ('a', 'b', 1, ('b', 'c'))
    actual = utils.parse_dims(dim, all_dims, replace_none=False)
    assert actual == expected