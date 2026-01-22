import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pyproj
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params

Tests for the Equidistant Conic coordinate system.

