import pytest
from bokeh.plotting import figure
from panel.layout import Row
from panel.links import Link
from panel.pane import Bokeh, HoloViews
from panel.tests.util import hv_available
from panel.widgets import (
def test_widget_link_target_param_not_found():
    t1 = TextInput()
    t2 = TextInput()
    with pytest.raises(ValueError) as excinfo:
        t1.jslink(t2, value='value1')
    assert "Could not jslink 'value1' parameter" in str(excinfo)