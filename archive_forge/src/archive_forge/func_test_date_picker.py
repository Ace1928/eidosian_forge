from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_date_picker(document, comm):
    date_picker = DatePicker(name='DatePicker', value=date(2018, 9, 2), start=date(2018, 9, 1), end=date(2018, 9, 10))
    widget = date_picker.get_root(document, comm=comm)
    assert isinstance(widget, date_picker._widget_type)
    assert widget.title == 'DatePicker'
    assert widget.value == '2018-09-02'
    assert widget.min_date == '2018-09-01'
    assert widget.max_date == '2018-09-10'
    widget.value = '2018-09-03'
    date_picker._process_events({'value': '2018-09-03'})
    assert date_picker.value == date(2018, 9, 3)
    date_picker._process_events({'value': date(2018, 9, 5)})
    assert date_picker.value == date(2018, 9, 5)
    date_picker._process_events({'value': date(2018, 9, 6)})
    assert date_picker.value == date(2018, 9, 6)
    date_picker.value = date(2018, 9, 4)
    assert widget.value == '2018-09-04'