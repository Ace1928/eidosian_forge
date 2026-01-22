import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
def test_patch_str():
    """
    Check that patches have nice and working `str` representation.

    Note that the logic is that `__str__` is defined such that:
    str(eval(str(p))) == str(p)
    """
    p = mpatches.Circle(xy=(1, 2), radius=3)
    assert str(p) == 'Circle(xy=(1, 2), radius=3)'
    p = mpatches.Ellipse(xy=(1, 2), width=3, height=4, angle=5)
    assert str(p) == 'Ellipse(xy=(1, 2), width=3, height=4, angle=5)'
    p = mpatches.Rectangle(xy=(1, 2), width=3, height=4, angle=5)
    assert str(p) == 'Rectangle(xy=(1, 2), width=3, height=4, angle=5)'
    p = mpatches.Wedge(center=(1, 2), r=3, theta1=4, theta2=5, width=6)
    assert str(p) == 'Wedge(center=(1, 2), r=3, theta1=4, theta2=5, width=6)'
    p = mpatches.Arc(xy=(1, 2), width=3, height=4, angle=5, theta1=6, theta2=7)
    expected = 'Arc(xy=(1, 2), width=3, height=4, angle=5, theta1=6, theta2=7)'
    assert str(p) == expected
    p = mpatches.Annulus(xy=(1, 2), r=(3, 4), width=1, angle=2)
    expected = 'Annulus(xy=(1, 2), r=(3, 4), width=1, angle=2)'
    assert str(p) == expected
    p = mpatches.RegularPolygon((1, 2), 20, radius=5)
    assert str(p) == 'RegularPolygon((1, 2), 20, radius=5, orientation=0)'
    p = mpatches.CirclePolygon(xy=(1, 2), radius=5, resolution=20)
    assert str(p) == 'CirclePolygon((1, 2), radius=5, resolution=20)'
    p = mpatches.FancyBboxPatch((1, 2), width=3, height=4)
    assert str(p) == 'FancyBboxPatch((1, 2), width=3, height=4)'
    path = mpath.Path([(1, 2), (2, 2), (1, 2)], closed=True)
    p = mpatches.PathPatch(path)
    assert str(p) == 'PathPatch3((1, 2) ...)'
    p = mpatches.Polygon(np.empty((0, 2)))
    assert str(p) == 'Polygon0()'
    data = [[1, 2], [2, 2], [1, 2]]
    p = mpatches.Polygon(data)
    assert str(p) == 'Polygon3((1, 2) ...)'
    p = mpatches.FancyArrowPatch(path=path)
    assert str(p)[:27] == 'FancyArrowPatch(Path(array('
    p = mpatches.FancyArrowPatch((1, 2), (3, 4))
    assert str(p) == 'FancyArrowPatch((1, 2)->(3, 4))'
    p = mpatches.ConnectionPatch((1, 2), (3, 4), 'data')
    assert str(p) == 'ConnectionPatch((1, 2), (3, 4))'
    s = mpatches.Shadow(p, 1, 1)
    assert str(s) == 'Shadow(ConnectionPatch((1, 2), (3, 4)))'