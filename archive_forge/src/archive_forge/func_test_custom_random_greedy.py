import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_custom_random_greedy():
    eq, shapes = oe.helpers.rand_equation(10, 4, seed=42)
    views = list(map(np.ones, shapes))
    with pytest.raises(ValueError):
        oe.RandomGreedy(minimize='something')
    optimizer = oe.RandomGreedy(max_repeats=10, minimize='flops')
    path, path_info = oe.contract_path(eq, *views, optimize=optimizer)
    assert len(optimizer.costs) == 10
    assert len(optimizer.sizes) == 10
    assert path == optimizer.path
    assert optimizer.best['flops'] == min(optimizer.costs)
    assert path_info.largest_intermediate == optimizer.best['size']
    assert path_info.opt_cost == optimizer.best['flops']
    optimizer.temperature = 0.0
    optimizer.max_repeats = 6
    path, path_info = oe.contract_path(eq, *views, optimize=optimizer)
    assert len(optimizer.costs) == 16
    assert len(optimizer.sizes) == 16
    assert path == optimizer.path
    assert optimizer.best['size'] == min(optimizer.sizes)
    assert path_info.largest_intermediate == optimizer.best['size']
    assert path_info.opt_cost == optimizer.best['flops']
    eq, shapes = oe.helpers.rand_equation(10, 4, seed=41)
    views = list(map(np.ones, shapes))
    with pytest.raises(ValueError):
        path, path_info = oe.contract_path(eq, *views, optimize=optimizer)