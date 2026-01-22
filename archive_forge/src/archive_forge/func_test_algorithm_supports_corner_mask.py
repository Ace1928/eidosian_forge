import datetime
import platform
import re
from unittest import mock
import contourpy
import numpy as np
from numpy.testing import (
import matplotlib as mpl
from matplotlib import pyplot as plt, rc_context, ticker
from matplotlib.colors import LogNorm, same_color
import matplotlib.patches as mpatches
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import pytest
@pytest.mark.parametrize('algorithm', ['mpl2005', 'mpl2014', 'serial', 'threaded'])
def test_algorithm_supports_corner_mask(algorithm):
    z = np.array([[1.0, 2.0], [3.0, 4.0]])
    plt.contourf(z, algorithm=algorithm, corner_mask=False)
    if algorithm != 'mpl2005':
        plt.contourf(z, algorithm=algorithm, corner_mask=True)
    else:
        with pytest.raises(ValueError):
            plt.contourf(z, algorithm=algorithm, corner_mask=True)