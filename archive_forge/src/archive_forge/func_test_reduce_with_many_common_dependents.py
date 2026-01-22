from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
@pytest.mark.parametrize('optimize', [True, False])
@pytest.mark.parametrize('keep_self', [True, False])
@pytest.mark.parametrize('ndeps', [2, 5])
@pytest.mark.parametrize('n_reducers', [4, 7])
def test_reduce_with_many_common_dependents(optimize, keep_self, ndeps, n_reducers):
    da = pytest.importorskip('dask.array')
    import numpy as np

    def random(**kwargs):
        assert len(kwargs) == ndeps
        return np.random.random((10, 10))
    trivial_deps = {f'k{i}': delayed(object(), name=f'object-{i}') for i in range(ndeps)}
    x = da.blockwise(random, 'yx', new_axes={'y': (10,) * n_reducers, 'x': (10,) * n_reducers}, dtype=float, **trivial_deps)
    graph = x.sum(axis=1, split_every=20)
    from dask.order import order
    if keep_self:
        dsk = collections_to_dsk([x, graph], optimize_graph=optimize)
    else:
        dsk = collections_to_dsk([graph], optimize_graph=optimize)
    dependencies, dependents = get_deps(dsk)
    before = len(dsk)
    before_dsk = dsk.copy()
    o = order(dsk)
    assert_topological_sort(dsk, o)
    assert before == len(o) == len(dsk)
    assert before_dsk == dsk
    assert {'object', 'sum', 'sum-aggregate'}.issubset({key_split(k) for k in o})
    reducers = {k for k in o if key_split(k) == 'sum-aggregate'}
    assert (p := max(diagnostics(dsk)[1])) <= n_reducers + ndeps, p
    if optimize:
        for r in reducers:
            prios_deps = []
            for dep in dependencies[r]:
                prios_deps.append(o[dep])
            assert max(prios_deps) - min(prios_deps) == (len(dependencies[r]) - 1) * (2 if keep_self else 1)