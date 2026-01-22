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
def test_corner_center():
    loc = [10, 20]
    width = 1
    height = 2
    corners = ((10, 20), (11, 20), (11, 22), (10, 22))
    rect = Rectangle(loc, width, height)
    assert_array_equal(rect.get_corners(), corners)
    assert_array_equal(rect.get_center(), (10.5, 21))
    corners_rot = ((10, 20), (10, 21), (8, 21), (8, 20))
    rect.set_angle(90)
    assert_array_equal(rect.get_corners(), corners_rot)
    assert_array_equal(rect.get_center(), (9, 20.5))
    theta = 33
    t = mtransforms.Affine2D().rotate_around(*loc, np.deg2rad(theta))
    corners_rot = t.transform(corners)
    rect.set_angle(theta)
    assert_almost_equal(rect.get_corners(), corners_rot)
    loc = [loc[0] + width / 2, loc[1] + height / 2]
    ellipse = Ellipse(loc, width, height)
    assert_array_equal(ellipse.get_corners(), corners)
    corners_rot = ((11.5, 20.5), (11.5, 21.5), (9.5, 21.5), (9.5, 20.5))
    ellipse.set_angle(90)
    assert_array_equal(ellipse.get_corners(), corners_rot)
    assert_array_equal(ellipse.get_center(), loc)
    theta = 33
    t = mtransforms.Affine2D().rotate_around(*loc, np.deg2rad(theta))
    corners_rot = t.transform(corners)
    ellipse.set_angle(theta)
    assert_almost_equal(ellipse.get_corners(), corners_rot)