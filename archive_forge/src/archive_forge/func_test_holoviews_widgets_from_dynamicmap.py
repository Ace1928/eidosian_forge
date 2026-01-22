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
def test_holoviews_widgets_from_dynamicmap(document, comm):
    range_dim = hv.Dimension('A', range=(0, 10.0))
    range_step_dim = hv.Dimension('B', range=(0, 10.0), step=0.2)
    range_default_dim = hv.Dimension('C', range=(0, 10.0), default=3)
    value_dim = hv.Dimension('D', values=['a', 'b', 'c'])
    value_default_dim = hv.Dimension('E', values=['a', 'b', 'c', 'd'], default='b')
    value_numeric_dim = hv.Dimension('F', values=[1, 3, 10], default=3)
    kdims = [range_dim, range_step_dim, range_default_dim, value_dim, value_default_dim, value_numeric_dim]
    dmap = hv.DynamicMap(lambda A, B, C, D, E, F: hv.Curve([]), kdims=kdims)
    widgets, _ = HoloViews.widgets_from_dimensions(dmap)
    assert len(widgets) == len(kdims)
    assert isinstance(widgets[0], FloatSlider)
    assert widgets[0].name == 'A'
    assert widgets[0].start == range_dim.range[0]
    assert widgets[0].end == range_dim.range[1]
    assert widgets[0].value == range_dim.range[0]
    assert widgets[0].step == 0.1
    assert isinstance(widgets[1], FloatSlider)
    assert widgets[1].name == 'B'
    assert widgets[1].start == range_step_dim.range[0]
    assert widgets[1].end == range_step_dim.range[1]
    assert widgets[1].value == range_step_dim.range[0]
    assert widgets[1].step == range_step_dim.step
    assert isinstance(widgets[2], FloatSlider)
    assert widgets[2].name == 'C'
    assert widgets[2].start == range_default_dim.range[0]
    assert widgets[2].end == range_default_dim.range[1]
    assert widgets[2].value == range_default_dim.default
    assert widgets[2].step == 0.1
    assert isinstance(widgets[3], Select)
    assert widgets[3].name == 'D'
    assert widgets[3].options == value_dim.values
    assert widgets[3].value == value_dim.values[0]
    assert isinstance(widgets[4], Select)
    assert widgets[4].name == 'E'
    assert widgets[4].options == value_default_dim.values
    assert widgets[4].value == value_default_dim.default
    assert isinstance(widgets[5], DiscreteSlider)
    assert widgets[5].name == 'F'
    assert widgets[5].options == {str(v): v for v in value_numeric_dim.values}
    assert widgets[5].value == value_numeric_dim.default