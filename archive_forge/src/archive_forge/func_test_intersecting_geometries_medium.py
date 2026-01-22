import pytest
import cartopy.feature as cfeature
def test_intersecting_geometries_medium(self, monkeypatch):
    monkeypatch.setattr(auto_land, 'geometries', lambda: [])
    auto_land.intersecting_geometries(medium_extent)
    assert auto_land.scale == '50m'