import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_extract_triangulation_positional_mask():
    mask = [True]
    args = [[0, 2, 1], [0, 0, 1], [[0, 1, 2]], mask]
    x_, y_, triangles_, mask_, args_, kwargs_ = mtri.Triangulation._extract_triangulation_params(args, {})
    assert mask_ is None
    assert args_ == [mask]