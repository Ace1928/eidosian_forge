from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
def test_datetime_range_slider(document, comm):
    datetime_slider = DatetimeRangeSlider(name='DatetimeRangeSlider', value=(datetime(2018, 9, 2), datetime(2018, 9, 4)), start=datetime(2018, 9, 1), end=datetime(2018, 9, 10))
    widget = datetime_slider.get_root(document, comm=comm)
    assert isinstance(widget, datetime_slider._widget_type)
    assert widget.title == 'DatetimeRangeSlider'
    assert widget.value == (1535846400000, 1536019200000)
    assert widget.start == 1535760000000
    assert widget.end == 1536537600000
    epoch = datetime(1970, 1, 1)
    widget.value = ((datetime(2018, 9, 3) - epoch).total_seconds() * 1000, (datetime(2018, 9, 6) - epoch).total_seconds() * 1000)
    datetime_slider._process_events({'value': widget.value})
    assert datetime_slider.value == (datetime(2018, 9, 3), datetime(2018, 9, 6))
    value_throttled = ((datetime(2018, 9, 3) - epoch).total_seconds() * 1000, (datetime(2018, 9, 6) - epoch).total_seconds() * 1000)
    datetime_slider._process_events({'value_throttled': value_throttled})
    assert datetime_slider.value == (datetime(2018, 9, 3), datetime(2018, 9, 6))
    datetime_slider.value = (datetime(2018, 9, 4), datetime(2018, 9, 6))
    assert widget.value == (1536019200000, 1536192000000)
    epoch_time = lambda dt: (dt - epoch).total_seconds() * 1000
    epoch_times = lambda *dts: tuple(map(epoch_time, dts))
    with config.set(throttled=True):
        datetime_slider._process_events({'value': epoch_times(datetime(2021, 2, 15), datetime(2021, 5, 15))})
        assert datetime_slider.value == (datetime(2018, 9, 4), datetime(2018, 9, 6))
        datetime_slider._process_events({'value_throttled': epoch_times(datetime(2021, 2, 15), datetime(2021, 5, 15))})
        assert datetime_slider.value == (datetime(2021, 2, 15), datetime(2021, 5, 15))
        datetime_slider.value = (datetime(2021, 2, 12), datetime(2021, 5, 12))
        assert widget.value == (1613088000000, 1620777600000)