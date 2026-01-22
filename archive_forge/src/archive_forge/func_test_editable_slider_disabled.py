from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
def test_editable_slider_disabled():
    slider = EditableFloatSlider(disabled=True)
    assert slider._slider.disabled
    assert slider._value_edit.disabled
    slider.disabled = False
    assert not slider._slider.disabled
    assert not slider._value_edit.disabled