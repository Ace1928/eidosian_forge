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
@pytest.mark.parametrize('algorithm, klass', [('mpl2005', contourpy.Mpl2005ContourGenerator), ('mpl2014', contourpy.Mpl2014ContourGenerator), ('serial', contourpy.SerialContourGenerator), ('threaded', contourpy.ThreadedContourGenerator), ('invalid', None)])
def test_algorithm_name(algorithm, klass):
    z = np.array([[1.0, 2.0], [3.0, 4.0]])
    if klass is not None:
        cs = plt.contourf(z, algorithm=algorithm)
        assert isinstance(cs._contour_generator, klass)
    else:
        with pytest.raises(ValueError):
            plt.contourf(z, algorithm=algorithm)