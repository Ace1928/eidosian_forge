import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
def test_footprint_octahedron(self):
    """Test octahedron footprints"""
    self.strel_worker_3d('data/diamond-matlab-output.npz', footprints.octahedron)