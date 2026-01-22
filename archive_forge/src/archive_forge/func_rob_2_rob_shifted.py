from unittest import mock
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot, InterProjectionTransform
def rob_2_rob_shifted(self):
    return InterProjectionTransform(ccrs.Robinson(), ccrs.Robinson(central_longitude=0))