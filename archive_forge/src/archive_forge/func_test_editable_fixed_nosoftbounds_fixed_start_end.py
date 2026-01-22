from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
@pytest.mark.parametrize('editableslider', [EditableFloatSlider, EditableIntSlider, EditableRangeSlider])
def test_editable_fixed_nosoftbounds_fixed_start_end(editableslider):
    start, end = (8, 9)
    fixed_start, fixed_end = (5, 10)
    slider = editableslider(fixed_start=fixed_start, fixed_end=fixed_end)
    assert slider.start == fixed_start
    assert slider.end == fixed_end
    slider = editableslider(fixed_start=fixed_start, fixed_end=fixed_end, start=start, end=end)
    assert slider.start == start
    assert slider.end == end