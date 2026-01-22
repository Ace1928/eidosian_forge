import datetime as dt
import warnings
import numpy as np
import pytest
from bokeh.models import (
from bokeh.plotting import figure
import panel as pn
from panel.depends import bind
from panel.layout import (
from panel.pane import HoloViews, PaneBase, panel
from panel.tests.util import hv_available, mpl_available
from panel.theme import Native
from panel.util.warnings import PanelDeprecationWarning
from panel.widgets import (
@hv_available
def test_holoviews_date_slider_widgets_from_holomap():
    hmap = hv.HoloMap({dt.datetime(2016, 1, i + 1): hv.Curve([i]) for i in range(3)}, kdims=['X'])
    widgets, _ = HoloViews.widgets_from_dimensions(hmap)
    assert isinstance(widgets[0], DiscreteSlider)
    assert widgets[0].name == 'X'
    assert widgets[0].options == {'2016-01-01 00:00:00': dt.datetime(2016, 1, 1), '2016-01-02 00:00:00': dt.datetime(2016, 1, 2), '2016-01-03 00:00:00': dt.datetime(2016, 1, 3)}
    assert widgets[0].value == dt.datetime(2016, 1, 1)