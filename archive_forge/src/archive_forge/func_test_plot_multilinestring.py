import pytest
from numpy.testing import assert_allclose
from shapely import box, get_coordinates, LineString, MultiLineString, Point
from shapely.plotting import patch_from_polygon, plot_line, plot_points, plot_polygon
def test_plot_multilinestring():
    line = MultiLineString([LineString([(0, 0), (1, 0), (1, 1)]), LineString([(2, 2), (3, 3)])])
    artist, _ = plot_line(line)
    plot_coords = artist.get_path().vertices
    assert_allclose(plot_coords, get_coordinates(line))