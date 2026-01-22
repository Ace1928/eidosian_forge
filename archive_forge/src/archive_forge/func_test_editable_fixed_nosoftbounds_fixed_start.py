from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
@pytest.mark.parametrize('editableslider', [EditableFloatSlider, EditableIntSlider, EditableRangeSlider])
def test_editable_fixed_nosoftbounds_fixed_start(editableslider):
    start, _ = (8, 9)
    fixed_start, _ = (5, 10)
    step = 2
    slider = editableslider(fixed_start=fixed_start)
    assert slider.start == fixed_start
    assert slider.end == fixed_start + 1
    slider = editableslider(fixed_start=fixed_start, step=step)
    assert slider.start == fixed_start
    assert slider.end == fixed_start + step
    slider = editableslider(fixed_start=fixed_start, start=start)
    assert slider.start == start
    assert slider.end == start + 1
    slider = editableslider(fixed_start=fixed_start, start=start, step=step)
    assert slider.start == start
    assert slider.end == start + step