import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def dipole_potential(x, y):
    """An electric dipole potential V."""
    r_sq = x ** 2 + y ** 2
    theta = np.arctan2(y, x)
    z = np.cos(theta) / r_sq
    return (np.max(z) - z) / (np.max(z) - np.min(z))