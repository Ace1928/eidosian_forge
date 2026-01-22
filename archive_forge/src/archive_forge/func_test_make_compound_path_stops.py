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
def test_make_compound_path_stops():
    zero = [0, 0]
    paths = 3 * [Path([zero, zero], [Path.MOVETO, Path.STOP])]
    compound_path = Path.make_compound_path(*paths)
    assert np.sum(compound_path.codes == Path.STOP) == 0