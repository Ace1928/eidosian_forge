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
@pytest.mark.parametrize('size', [3, 4, 5, 10])
@pytest.mark.parametrize('backend', backends)
def test_chain_2(size, backend):
    xs = [np.random.rand(2, 2) for _ in range(size)]
    shapes = [x.shape for x in xs]
    alphabet = ''.join((get_symbol(i) for i in range(size + 1)))
    names = [alphabet[i:i + 2] for i in range(size)]
    inputs = ','.join(names)
    with shared_intermediates():
        print(inputs)
        for i in range(size):
            target = alphabet[i:i + 2]
            eq = '{}->{}'.format(inputs, target)
            path_info = contract_path(eq, *xs)
            print(path_info[1])
            expr = contract_expression(eq, *shapes)
            expr(*xs, backend=backend)
        print('-' * 40)