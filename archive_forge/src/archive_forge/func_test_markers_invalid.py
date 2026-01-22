import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
@pytest.mark.parametrize('marker', ['square', np.array([[-0.5, 0, 1, 2, 3]]), (1,), (5, 3), (1, 2, 3, 4)])
def test_markers_invalid(marker):
    with pytest.raises(ValueError):
        markers.MarkerStyle(marker)