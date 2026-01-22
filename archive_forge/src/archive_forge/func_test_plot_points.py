import pytest
from numpy.testing import assert_allclose
from shapely import box, get_coordinates, LineString, MultiLineString, Point
from shapely.plotting import patch_from_polygon, plot_line, plot_points, plot_polygon
def test_plot_points():
    for geom in [Point(0, 0), LineString([(0, 0), (1, 0), (1, 1)]), box(0, 0, 1, 1)]:
        artist = plot_points(geom)
        plot_coords = artist.get_path().vertices
        assert_allclose(plot_coords, get_coordinates(geom))
        assert artist.get_linestyle() == 'None'
    geom = Point(0, 0)
    artist = plot_points(geom, color='red', marker='+', fillstyle='top')
    assert artist.get_color() == 'red'
    assert artist.get_marker() == '+'
    assert artist.get_fillstyle() == 'top'