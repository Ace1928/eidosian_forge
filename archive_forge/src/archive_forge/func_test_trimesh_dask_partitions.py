from __future__ import annotations
import os
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask.context import config
from numpy import nan
import datashader as ds
from datashader.datatypes import RaggedArray
import datashader.utils as du
import pytest
from datashader.tests.test_pandas import (
@pytest.mark.parametrize('npartitions', list(range(1, 6)))
def test_trimesh_dask_partitions(npartitions):
    """Assert that when two triangles share an edge that would normally get
    double-drawn, the edge is only drawn for the rightmost (or bottommost)
    triangle.
    """
    verts = dd.from_pandas(pd.DataFrame({'x': [4, 1, 5, 5, 5, 4], 'y': [4, 5, 5, 5, 4, 4]}), npartitions=npartitions)
    tris = dd.from_pandas(pd.DataFrame({'v0': [0, 3], 'v1': [1, 4], 'v2': [2, 5], 'val': [1, 2]}), npartitions=npartitions)
    cvs = ds.Canvas(plot_width=20, plot_height=20, x_range=(0, 5), y_range=(0, 5))
    mesh = du.mesh(verts, tris)
    assert isinstance(mesh, dd.DataFrame)
    n = len(mesh)
    assert n == 6
    expected_chunksize = int(np.ceil(len(mesh) / (3 * npartitions)) * 3)
    expected_npartitions = int(np.ceil(n / expected_chunksize))
    assert expected_npartitions == mesh.npartitions
    partitions_lens = mesh.map_partitions(len).compute()
    for partitions_len in partitions_lens:
        assert partitions_len % 3 == 0
    agg = cvs.trimesh(verts, tris, mesh)
    sol = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='i4')
    np.testing.assert_array_equal(np.flipud(agg.fillna(0).astype('i4').values)[:5], sol)
    assert_eq_ndarray(agg.x_range, (0, 5), close=True)
    assert_eq_ndarray(agg.y_range, (0, 5), close=True)