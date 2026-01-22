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
def test_path_deepcopy():
    verts = [[0, 0], [1, 1]]
    codes = [Path.MOVETO, Path.LINETO]
    path1 = Path(verts)
    path2 = Path(verts, codes)
    path1_copy = path1.deepcopy()
    path2_copy = path2.deepcopy()
    assert path1 is not path1_copy
    assert path1.vertices is not path1_copy.vertices
    assert path2 is not path2_copy
    assert path2.vertices is not path2_copy.vertices
    assert path2.codes is not path2_copy.codes