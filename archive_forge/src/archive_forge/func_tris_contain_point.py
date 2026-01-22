import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def tris_contain_point(triang, xy):
    return sum((tri_contains_point(triang.x[tri], triang.y[tri], xy) for tri in triang.triangles))