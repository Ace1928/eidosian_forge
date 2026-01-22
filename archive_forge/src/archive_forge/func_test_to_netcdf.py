from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from xarray.backends.api import open_datatree
from xarray.datatree_.datatree.testing import assert_equal
from xarray.tests import (
def test_to_netcdf(self, tmpdir, simple_datatree):
    filepath = tmpdir / 'test.nc'
    original_dt = simple_datatree
    original_dt.to_netcdf(filepath, engine=self.engine)
    roundtrip_dt = open_datatree(filepath, engine=self.engine)
    assert_equal(original_dt, roundtrip_dt)