import pytest
import cartopy.feature as cfeature
def test_intersecting_geometries_large(self, monkeypatch):
    monkeypatch.setattr(auto_land, 'geometries', lambda: [])
    auto_land.intersecting_geometries(large_extent)
    assert auto_land.scale == '110m'