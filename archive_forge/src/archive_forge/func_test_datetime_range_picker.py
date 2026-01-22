from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_datetime_range_picker(document, comm):
    datetime_range_picker = DatetimeRangePicker(name='DatetimeRangePicker', value=(datetime(2018, 9, 2, 1, 5), datetime(2018, 9, 2, 1, 6)), start=date(2018, 9, 1), end=datetime(2018, 9, 10))
    widget = datetime_range_picker.get_root(document, comm=comm)
    assert isinstance(widget, datetime_range_picker._widget_type)
    assert widget.title == 'DatetimeRangePicker'
    assert widget.value == '2018-09-02 01:05:00 to 2018-09-02 01:06:00'
    assert widget.min_date == '2018-09-01T00:00:00'
    assert widget.max_date == '2018-09-10T00:00:00'
    datetime_range_picker._process_events({'value': '2018-09-03 03:00:01 to 2018-09-04 03:00:01'})
    assert datetime_range_picker.value == (datetime(2018, 9, 3, 3, 0, 1), datetime(2018, 9, 4, 3, 0, 1))
    value = datetime_range_picker._process_param_change({'value': (datetime(2018, 9, 4, 1, 0, 1), datetime(2018, 9, 4, 4, 0, 1))})
    assert value['value'] == '2018-09-04 01:00:01 to 2018-09-04 04:00:01'
    value = (datetime(2018, 9, 4, 12, 1), datetime(2018, 9, 4, 12, 1, 10))
    assert datetime_range_picker._deserialize_value(value) == '2018-09-04 12:01:00 to 2018-09-04 12:01:10'
    assert datetime_range_picker._serialize_value(datetime_range_picker._deserialize_value(value)) == value
    with pytest.raises(ValueError):
        datetime_range_picker._process_events({'value': '2018-08-31 23:59:59'})
    with pytest.raises(ValueError):
        datetime_range_picker._process_events({'value': '2018-09-10 00:00:01'})