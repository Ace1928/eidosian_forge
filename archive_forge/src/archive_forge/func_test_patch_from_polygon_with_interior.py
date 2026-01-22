import pytest
from numpy.testing import assert_allclose
from shapely import box, get_coordinates, LineString, MultiLineString, Point
from shapely.plotting import patch_from_polygon, plot_line, plot_points, plot_polygon
def test_patch_from_polygon_with_interior():
    poly = box(0, 0, 1, 1).difference(box(0.2, 0.2, 0.5, 0.5))
    artist = patch_from_polygon(poly, facecolor='red', edgecolor='blue', linewidth=3)
    assert equal_color(artist.get_facecolor(), 'red')
    assert equal_color(artist.get_edgecolor(), 'blue')
    assert artist.get_linewidth() == 3