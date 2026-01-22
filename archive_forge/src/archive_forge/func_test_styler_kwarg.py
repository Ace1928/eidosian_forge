from unittest import mock
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot, InterProjectionTransform
@mock.patch('cartopy.mpl.geoaxes.GeoAxes.add_feature')
@mock.patch('cartopy.feature.ShapelyFeature')
def test_styler_kwarg(self, ShapelyFeature, add_feature_method):
    ax = GeoAxes(plt.figure(), [0, 0, 1, 1], projection=ccrs.Robinson())
    ax.add_geometries(mock.sentinel.geometries, mock.sentinel.crs, styler=mock.sentinel.styler, wibble='wobble')
    ShapelyFeature.assert_called_once_with(mock.sentinel.geometries, mock.sentinel.crs, wibble='wobble')
    add_feature_method.assert_called_once_with(ShapelyFeature(), styler=mock.sentinel.styler)