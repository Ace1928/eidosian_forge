from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
@pytest.mark.parametrize('editableslider,start,end,fixed_start,fixed_end,val_init,val_update,fail_init,fail_update', [(EditableRangeSlider, 1, 5, 0, None, (2, 5), (3, 5), False, False), (EditableRangeSlider, 1, 5, 0, None, (2, 5), (-1, 4), False, True), (EditableRangeSlider, 1, 5, 0, None, (-1, 100), (-1, 200), True, True), (EditableRangeSlider, 1, 5, 0, None, (1, 5), (0, 100), False, False)])
def test_editable_rangeslider_bounds(editableslider, start, end, fixed_start, fixed_end, val_init, val_update, fail_init, fail_update):
    try:
        slider = editableslider(start=start, end=end, fixed_start=fixed_start, fixed_end=fixed_end, value=val_init, name='Slider')
    except Exception:
        assert fail_init
    try:
        slider.value = val_update
    except Exception:
        assert fail_update