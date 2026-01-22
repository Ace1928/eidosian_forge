import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
import shapely.geometry as sgeom
from cartopy import geodesic
def test_geometry_length_point():
    geod = geodesic.Geodesic()
    geom = sgeom.Point(lhr)
    with pytest.raises(TypeError):
        geod.geometry_length(geom)