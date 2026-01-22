from __future__ import annotations
import pytest
from xarray import DataArray, tutorial
from xarray.tests import assert_identical, network
def test_download_from_github(self, tmp_path) -> None:
    cache_dir = tmp_path / tutorial._default_cache_dir_name
    ds = tutorial.open_dataset(self.testfile, cache_dir=cache_dir).load()
    tiny = DataArray(range(5), name='tiny').to_dataset()
    assert_identical(ds, tiny)