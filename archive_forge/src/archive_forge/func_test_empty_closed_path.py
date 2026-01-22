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
def test_empty_closed_path():
    path = Path(np.zeros((0, 2)), closed=True)
    assert path.vertices.shape == (0, 2)
    assert path.codes is None
    assert_array_equal(path.get_extents().extents, transforms.Bbox.null().extents)