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
def test_holoviews_widgets_from_holomap():
    hmap = hv.HoloMap({(i, chr(65 + i)): hv.Curve([i]) for i in range(3)}, kdims=['X', 'Y'])
    widgets, _ = HoloViews.widgets_from_dimensions(hmap)
    assert isinstance(widgets[0], DiscreteSlider)
    assert widgets[0].name == 'X'
    assert widgets[0].options == {str(i): i for i in range(3)}
    assert widgets[0].value == 0
    assert isinstance(widgets[1], Select)
    assert widgets[1].name == 'Y'
    assert widgets[1].options == ['A', 'B', 'C']
    assert widgets[1].value == 'A'