from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
@pytest.mark.parametrize('editableslider,fixed_start,fixed_end', [(EditableFloatSlider, 5, 10), (EditableIntSlider, 5, 10)])
def test_editable_slider_fixed_novalue(editableslider, fixed_start, fixed_end):
    slider = editableslider(fixed_start=fixed_start, fixed_end=fixed_end)
    assert slider.value == fixed_start
    slider = editableslider(fixed_start=fixed_start)
    assert slider.value == fixed_start
    slider = editableslider(fixed_end=fixed_end)
    assert slider.value == fixed_end - 1