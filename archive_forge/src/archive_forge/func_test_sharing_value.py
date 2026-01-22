import itertools
import weakref
from collections import Counter
import numpy as np
import pytest
from opt_einsum import (contract, contract_expression, contract_path, get_symbol, helpers, shared_intermediates)
from opt_einsum.backends import to_cupy, to_torch
from opt_einsum.contract import _einsum
from opt_einsum.parser import parse_einsum_input
from opt_einsum.sharing import (count_cached_ops, currently_sharing, get_sharing_cache)
@pytest.mark.parametrize('eq', equations)
@pytest.mark.parametrize('backend', backends)
def test_sharing_value(eq, backend):
    views = helpers.build_views(eq)
    shapes = [v.shape for v in views]
    expr = contract_expression(eq, *shapes)
    expected = expr(*views, backend=backend)
    with shared_intermediates():
        actual = expr(*views, backend=backend)
    assert (actual == expected).all()