import pytest
import cartopy.feature as cfeature
def test_intersecting_geometries_small(self, monkeypatch):
    monkeypatch.setattr(auto_land, 'geometries', lambda: [])
    auto_land.intersecting_geometries(small_extent)
    assert auto_land.scale == '10m'