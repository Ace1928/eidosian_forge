import re
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.backend_bases import MouseEvent
def test_path_exceptions():
    bad_verts1 = np.arange(12).reshape(4, 3)
    with pytest.raises(ValueError, match=re.escape(f'has shape {bad_verts1.shape}')):
        Path(bad_verts1)
    bad_verts2 = np.arange(12).reshape(2, 3, 2)
    with pytest.raises(ValueError, match=re.escape(f'has shape {bad_verts2.shape}')):
        Path(bad_verts2)
    good_verts = np.arange(12).reshape(6, 2)
    bad_codes = np.arange(2)
    msg = re.escape(f'Your vertices have shape {good_verts.shape} but your codes have shape {bad_codes.shape}')
    with pytest.raises(ValueError, match=msg):
        Path(good_verts, bad_codes)