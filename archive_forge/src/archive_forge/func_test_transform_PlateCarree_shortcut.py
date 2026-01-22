from unittest import mock
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot, InterProjectionTransform
def test_transform_PlateCarree_shortcut():
    src = ccrs.PlateCarree(central_longitude=0)
    target = ccrs.PlateCarree(central_longitude=180)
    pth1 = mpath.Path([[0.5, 0], [10, 10]])
    pth2 = mpath.Path([[0.5, 91], [10, 10]])
    pth3 = mpath.Path([[-0.5, 0], [10, 10]])
    trans = InterProjectionTransform(src, target)
    with mock.patch.object(target, 'project_geometry', wraps=target.project_geometry) as counter:
        trans.transform_path(pth1)
        counter.assert_not_called()
    with mock.patch.object(target, 'project_geometry', wraps=target.project_geometry) as counter:
        trans.transform_path(pth2)
        counter.assert_called_once()
    with mock.patch.object(target, 'project_geometry', wraps=target.project_geometry) as counter:
        trans.transform_path(pth3)
        counter.assert_called_once()