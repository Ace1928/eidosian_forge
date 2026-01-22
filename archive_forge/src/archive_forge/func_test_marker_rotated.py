import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
@pytest.mark.parametrize('marker,deg,rad,expected', [(markers.MarkerStyle('o'), 10, None, Affine2D().rotate_deg(10)), (markers.MarkerStyle('o'), None, 0.01, Affine2D().rotate(0.01)), (markers.MarkerStyle('o', transform=Affine2D().translate(1, 1)), 10, None, Affine2D().translate(1, 1).rotate_deg(10)), (markers.MarkerStyle('o', transform=Affine2D().translate(1, 1)), None, 0.01, Affine2D().translate(1, 1).rotate(0.01)), (markers.MarkerStyle('$|||$', transform=Affine2D().translate(1, 1)), 10, None, Affine2D().translate(1, 1).rotate_deg(10)), (markers.MarkerStyle(markers.TICKLEFT, transform=Affine2D().translate(1, 1)), 10, None, Affine2D().translate(1, 1).rotate_deg(10))])
def test_marker_rotated(marker, deg, rad, expected):
    new_marker = marker.rotated(deg=deg, rad=rad)
    assert new_marker is not marker
    assert new_marker.get_user_transform() == expected
    assert marker._user_transform is not new_marker._user_transform