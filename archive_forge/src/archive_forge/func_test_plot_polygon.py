import pytest
from numpy.testing import assert_allclose
from shapely import box, get_coordinates, LineString, MultiLineString, Point
from shapely.plotting import patch_from_polygon, plot_line, plot_points, plot_polygon
def test_plot_polygon():
    poly = box(0, 0, 1, 1)
    artist, _ = plot_polygon(poly)
    plot_coords = artist.get_path().vertices
    assert_allclose(plot_coords, get_coordinates(poly))
    artist = plot_polygon(poly, add_points=False, color='red', linewidth=3)
    assert equal_color(artist.get_facecolor(), 'red', alpha=0.3)
    assert equal_color(artist.get_edgecolor(), 'red', alpha=1.0)
    assert artist.get_linewidth() == 3