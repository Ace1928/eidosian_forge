import numpy as np
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
import pytest
@pytest.mark.parametrize('marker,transform,expected', [(markers.MarkerStyle('o'), Affine2D().translate(1, 1), Affine2D().translate(1, 1)), (markers.MarkerStyle('o', transform=Affine2D().translate(1, 1)), Affine2D().translate(1, 1), Affine2D().translate(2, 2)), (markers.MarkerStyle('$|||$', transform=Affine2D().translate(1, 1)), Affine2D().translate(1, 1), Affine2D().translate(2, 2)), (markers.MarkerStyle(markers.TICKLEFT, transform=Affine2D().translate(1, 1)), Affine2D().translate(1, 1), Affine2D().translate(2, 2))])
def test_marker_transformed(marker, transform, expected):
    new_marker = marker.transformed(transform)
    assert new_marker is not marker
    assert new_marker.get_user_transform() == expected
    assert marker._user_transform is not new_marker._user_transform