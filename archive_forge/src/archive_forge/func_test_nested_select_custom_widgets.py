import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_custom_widgets(document, comm):
    options = {'Andrew': {'temp': [1000, 925, 700, 500, 300], 'vorticity': [500, 300]}, 'Ben': {'temp': [500, 300], 'windspeed': [700, 500, 300]}}
    select = NestedSelect(options=options, levels=[{'name': 'Name', 'type': Select, 'width': 250}, {'name': 'Variable', 'type': Select}, {'name': 'lvl', 'type': DiscreteSlider}])
    widget_0 = select._widgets[0]
    widget_1 = select._widgets[1]
    widget_2 = select._widgets[2]
    assert isinstance(widget_0, Select)
    assert isinstance(widget_1, Select)
    assert isinstance(widget_2, DiscreteSlider)
    assert widget_0.width == 250
    assert widget_0.name == 'Name'
    assert widget_1.name == 'Variable'
    assert widget_2.name == 'lvl'
    assert widget_0.options == ['Andrew', 'Ben']
    assert widget_1.options == ['temp', 'vorticity']
    assert widget_2.options == [1000, 925, 700, 500, 300]