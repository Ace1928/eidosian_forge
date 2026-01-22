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
def test_polygon_selector_clear_method(ax):
    onselect = mock.Mock(spec=noop, return_value=None)
    tool = widgets.PolygonSelector(ax, onselect)
    for result in ([(50, 50), (150, 50), (50, 150), (50, 50)], [(50, 50), (100, 50), (50, 150), (50, 50)]):
        for x, y in result:
            for etype, event_args in polygon_place_vertex(x, y):
                do_event(tool, etype, **event_args)
        artist = tool._selection_artist
        assert tool._selection_completed
        assert tool.get_visible()
        assert artist.get_visible()
        np.testing.assert_equal(artist.get_xydata(), result)
        assert onselect.call_args == ((result[:-1],), {})
        tool.clear()
        assert not tool._selection_completed
        np.testing.assert_equal(artist.get_xydata(), [(0, 0)])