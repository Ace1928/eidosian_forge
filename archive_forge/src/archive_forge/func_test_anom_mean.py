from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_anom_mean():
    np = pytest.importorskip('numpy')
    xr = pytest.importorskip('xarray')
    import dask.array as da
    data = da.random.random((200, 1), chunks=(1, -1))
    ngroups = 5
    arr = xr.DataArray(data, dims=['time', 'x'], coords={'day': ('time', np.arange(data.shape[0]) % ngroups)})
    clim = arr.groupby('day').mean(dim='time')
    anom = arr.groupby('day') - clim
    anom_mean = anom.mean(dim='time')
    graph = collections_to_dsk([anom_mean])
    dependencies, dependents = get_deps(graph)
    diags, pressure = diagnostics(graph)
    assert max(pressure) <= 177
    from collections import defaultdict
    count_dependents = defaultdict(set)
    for k in dict(graph).keys():
        count_dependents[len(dependents[k])].add(k)
    n_splits = max(count_dependents)
    assert n_splits > 30
    transpose_tasks = count_dependents[n_splits]
    transpose_metrics = {k: diags[k] for k in transpose_tasks}
    assert len(transpose_metrics) == ngroups, {key_split(k) for k in diags}
    ages_mean_chunks = {k: v.age for k, v in diags.items() if 'mean_chunk' in k[0]}
    avg_age_mean_chunks = sum(ages_mean_chunks.values()) / len(ages_mean_chunks)
    max_age_mean_chunks = max(ages_mean_chunks.values())
    ages_tranpose = {k: v.age for k, v in transpose_metrics.items()}
    assert max_age_mean_chunks > 900
    assert avg_age_mean_chunks > 100
    avg_age_transpose = sum(ages_tranpose.values()) / len(ages_tranpose)
    max_age_transpose = max(ages_tranpose.values())
    assert max_age_transpose < 150
    assert avg_age_transpose < 100
    assert sum(pressure) / len(pressure) < 100