import functools
import itertools
import platform
import pytest
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib.backend_bases import (MouseButton, MouseEvent,
from matplotlib import cm
from matplotlib import colors as mcolors, patches as mpatch
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.testing.widgets import mock_event
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
@pytest.mark.parametrize('value', [np.inf, np.nan])
@pytest.mark.parametrize(('setter', 'side'), [('set_xlim3d', 'left'), ('set_xlim3d', 'right'), ('set_ylim3d', 'bottom'), ('set_ylim3d', 'top'), ('set_zlim3d', 'bottom'), ('set_zlim3d', 'top')])
def test_invalid_axes_limits(setter, side, value):
    limit = {side: value}
    fig = plt.figure()
    obj = fig.add_subplot(projection='3d')
    with pytest.raises(ValueError):
        getattr(obj, setter)(**limit)