import functools
import io
from unittest import mock
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
import matplotlib.colors as mcolors
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing.widgets import (click_and_drag, do_event, get_ax,
import numpy as np
from numpy.testing import assert_allclose
import pytest
@pytest.mark.parametrize('draw_bounding_box', [False, True])
def test_polygon_selector_set_props_handle_props(ax, draw_bounding_box):
    tool = widgets.PolygonSelector(ax, onselect=noop, props=dict(color='b', alpha=0.2), handle_props=dict(alpha=0.5), draw_bounding_box=draw_bounding_box)
    event_sequence = [*polygon_place_vertex(50, 50), *polygon_place_vertex(150, 50), *polygon_place_vertex(50, 150), *polygon_place_vertex(50, 50)]
    for etype, event_args in event_sequence:
        do_event(tool, etype, **event_args)
    artist = tool._selection_artist
    assert artist.get_color() == 'b'
    assert artist.get_alpha() == 0.2
    tool.set_props(color='r', alpha=0.3)
    assert artist.get_color() == 'r'
    assert artist.get_alpha() == 0.3
    for artist in tool._handles_artists:
        assert artist.get_color() == 'b'
        assert artist.get_alpha() == 0.5
    tool.set_handle_props(color='r', alpha=0.3)
    for artist in tool._handles_artists:
        assert artist.get_color() == 'r'
        assert artist.get_alpha() == 0.3