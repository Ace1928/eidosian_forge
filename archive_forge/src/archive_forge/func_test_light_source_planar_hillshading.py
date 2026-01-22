import copy
import itertools
import unittest.mock
from packaging.version import parse as parse_version
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
import base64
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib import cbook, cm
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.rcsetup import cycler
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_light_source_planar_hillshading():
    """
    Ensure that the illumination intensity is correct for planar surfaces.
    """

    def plane(azimuth, elevation, x, y):
        """
        Create a plane whose normal vector is at the given azimuth and
        elevation.
        """
        theta, phi = _azimuth2math(azimuth, elevation)
        a, b, c = _sph2cart(theta, phi)
        z = -(a * x + b * y) / c
        return z

    def angled_plane(azimuth, elevation, angle, x, y):
        """
        Create a plane whose normal vector is at an angle from the given
        azimuth and elevation.
        """
        elevation = elevation + angle
        if elevation > 90:
            azimuth = (azimuth + 180) % 360
            elevation = (90 - elevation) % 90
        return plane(azimuth, elevation, x, y)
    y, x = np.mgrid[5:0:-1, :5]
    for az, elev in itertools.product(range(0, 390, 30), range(0, 105, 15)):
        ls = mcolors.LightSource(az, elev)
        for angle in range(0, 105, 15):
            z = angled_plane(az, elev, angle, x, y)
            h = ls.hillshade(z)
            assert_array_almost_equal(h, np.cos(np.radians(angle)))