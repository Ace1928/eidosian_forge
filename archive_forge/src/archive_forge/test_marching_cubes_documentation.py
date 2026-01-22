import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import marching_cubes, mesh_surface_area
Compare two meshes, using a certain tolerance and invariant to
    the order of the faces.
    