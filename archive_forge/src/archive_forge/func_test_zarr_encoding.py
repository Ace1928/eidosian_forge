from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from xarray.backends.api import open_datatree
from xarray.datatree_.datatree.testing import assert_equal
from xarray.tests import (
def test_zarr_encoding(self, tmpdir, simple_datatree):
    import zarr
    filepath = tmpdir / 'test.zarr'
    original_dt = simple_datatree
    comp = {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)}
    enc = {'/set2': {var: comp for var in original_dt['/set2'].ds.data_vars}}
    original_dt.to_zarr(filepath, encoding=enc)
    roundtrip_dt = open_datatree(filepath, engine='zarr')
    print(roundtrip_dt['/set2/a'].encoding)
    assert roundtrip_dt['/set2/a'].encoding['compressor'] == comp['compressor']
    enc['/not/a/group'] = {'foo': 'bar'}
    with pytest.raises(ValueError, match='unexpected encoding group.*'):
        original_dt.to_zarr(filepath, encoding=enc, engine='zarr')