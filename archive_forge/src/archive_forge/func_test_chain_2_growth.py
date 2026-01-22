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
def test_chain_2_growth(backend):
    sizes = list(range(1, 21))
    costs = []
    for size in sizes:
        xs = [np.random.rand(2, 2) for _ in range(size)]
        alphabet = ''.join((get_symbol(i) for i in range(size + 1)))
        names = [alphabet[i:i + 2] for i in range(size)]
        inputs = ','.join(names)
        with shared_intermediates() as cache:
            for i in range(size):
                target = alphabet[i:i + 2]
                eq = '{}->{}'.format(inputs, target)
                expr = contract_expression(eq, *(x.shape for x in xs))
                expr(*xs, backend=backend)
            costs.append(_compute_cost(cache))
    print('sizes = {}'.format(repr(sizes)))
    print('costs = {}'.format(repr(costs)))
    for size, cost in zip(sizes, costs):
        print('{}\t{}'.format(size, cost))