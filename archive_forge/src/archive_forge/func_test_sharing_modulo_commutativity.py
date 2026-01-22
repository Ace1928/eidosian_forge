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
def test_sharing_modulo_commutativity(eq, backend):
    ops = helpers.build_views(eq)
    ops = [to_backend[backend](x) for x in ops]
    inputs, output, _ = parse_einsum_input([eq] + ops)
    inputs = inputs.split(',')
    print('-' * 40)
    print('Without sharing:')
    with shared_intermediates() as cache:
        _einsum(eq, *ops, backend=backend)
        expected = count_cached_ops(cache)
    print('-' * 40)
    print('With sharing:')
    with shared_intermediates() as cache:
        for permuted in itertools.permutations(zip(inputs, ops)):
            permuted_inputs = [p[0] for p in permuted]
            permuted_ops = [p[1] for p in permuted]
            permuted_eq = '{}->{}'.format(','.join(permuted_inputs), output)
            _einsum(permuted_eq, *permuted_ops, backend=backend)
        actual = count_cached_ops(cache)
    print('-' * 40)
    print('Without sharing: {} expressions'.format(expected))
    print('With sharing: {} expressions'.format(actual))
    assert actual == expected