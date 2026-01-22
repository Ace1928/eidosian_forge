import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
@pytest.mark.skipif(sys.version_info < (3, 2), reason='requires python3.2 or higher')
def test_parallel_random_greedy():
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(2)
    eq, shapes = oe.helpers.rand_equation(10, 4, seed=42)
    views = list(map(np.ones, shapes))
    optimizer = oe.RandomGreedy(max_repeats=10, parallel=pool)
    path, path_info = oe.contract_path(eq, *views, optimize=optimizer)
    assert len(optimizer.costs) == 10
    assert len(optimizer.sizes) == 10
    assert path == optimizer.path
    assert optimizer.parallel is pool
    assert optimizer._executor is pool
    assert optimizer.best['flops'] == min(optimizer.costs)
    assert path_info.largest_intermediate == optimizer.best['size']
    assert path_info.opt_cost == optimizer.best['flops']
    optimizer.max_repeats = int(1000000.0)
    optimizer.max_time = 0.2
    optimizer.parallel = 2
    path, path_info = oe.contract_path(eq, *views, optimize=optimizer)
    assert len(optimizer.costs) > 10
    assert len(optimizer.sizes) > 10
    assert path == optimizer.path
    assert optimizer.best['flops'] == min(optimizer.costs)
    assert path_info.largest_intermediate == optimizer.best['size']
    assert path_info.opt_cost == optimizer.best['flops']
    optimizer.parallel = True
    assert optimizer._executor is not None
    assert optimizer._executor is not pool
    are_done = [f.running() or f.done() for f in optimizer._futures]
    assert all(are_done)