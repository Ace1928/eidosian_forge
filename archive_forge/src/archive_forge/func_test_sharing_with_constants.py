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
@pytest.mark.parametrize('backend', backends)
def test_sharing_with_constants(backend):
    inputs = 'ij,jk,kl'
    outputs = 'ijkl'
    equations = ['{}->{}'.format(inputs, output) for output in outputs]
    shapes = ((2, 3), (3, 4), (4, 5))
    constants = {0, 2}
    ops = [np.random.rand(*shp) if i in constants else shp for i, shp in enumerate(shapes)]
    var = np.random.rand(*shapes[1])
    expected = [contract_expression(eq, *shapes)(ops[0], var, ops[2]) for eq in equations]
    with shared_intermediates():
        actual = [contract_expression(eq, *ops, constants=constants)(var) for eq in equations]
    for dim, expected_dim, actual_dim in zip(outputs, expected, actual):
        assert np.allclose(expected_dim, actual_dim), 'error at {}'.format(dim)