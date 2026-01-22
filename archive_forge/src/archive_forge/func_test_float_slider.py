from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
def test_float_slider(document, comm):
    slider = FloatSlider(start=0.1, end=0.5, value=0.4, name='Slider')
    widget = slider.get_root(document, comm=comm)
    assert isinstance(widget, slider._widget_type)
    assert widget.title == 'Slider'
    assert widget.step == 0.1
    assert widget.start == 0.1
    assert widget.end == 0.5
    assert widget.value == 0.4
    slider._process_events({'value': 0.2})
    assert slider.value == 0.2
    slider._process_events({'value_throttled': 0.2})
    assert slider.value_throttled == 0.2
    slider.value = 0.3
    assert widget.value == 0.3
    with config.set(throttled=True):
        slider._process_events({'value': 0.4})
        assert slider.value == 0.3
        slider._process_events({'value_throttled': 0.4})
        assert slider.value == 0.4
        slider.value = 0.5
        assert widget.value == 0.5