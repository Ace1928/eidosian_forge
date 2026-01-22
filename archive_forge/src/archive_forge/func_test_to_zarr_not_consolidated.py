from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from xarray.backends.api import open_datatree
from xarray.datatree_.datatree.testing import assert_equal
from xarray.tests import (
def test_to_zarr_not_consolidated(self, tmpdir, simple_datatree):
    filepath = tmpdir / 'test.zarr'
    zmetadata = filepath / '.zmetadata'
    s1zmetadata = filepath / 'set1' / '.zmetadata'
    filepath = str(filepath)
    original_dt = simple_datatree
    original_dt.to_zarr(filepath, consolidated=False)
    assert not zmetadata.exists()
    assert not s1zmetadata.exists()
    with pytest.warns(RuntimeWarning, match='consolidated'):
        roundtrip_dt = open_datatree(filepath, engine='zarr')
    assert_equal(original_dt, roundtrip_dt)