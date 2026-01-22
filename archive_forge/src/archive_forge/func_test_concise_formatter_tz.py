import datetime
import dateutil.tz
import dateutil.rrule
import functools
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import rc_context, style
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.ticker as mticker
def test_concise_formatter_tz():

    def _create_auto_date_locator(date1, date2, tz):
        fig, ax = plt.subplots()
        locator = mdates.AutoDateLocator(interval_multiples=True)
        formatter = mdates.ConciseDateFormatter(locator, tz=tz)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(date1, date2)
        fig.canvas.draw()
        sts = [st.get_text() for st in ax.get_yticklabels()]
        return (sts, ax.yaxis.get_offset_text().get_text())
    d1 = datetime.datetime(1997, 1, 1).replace(tzinfo=datetime.timezone.utc)
    results = ([datetime.timedelta(hours=40), ['03:00', '07:00', '11:00', '15:00', '19:00', '23:00', '03:00', '07:00', '11:00', '15:00', '19:00'], '1997-Jan-02'], [datetime.timedelta(minutes=20), ['03:00', '03:05', '03:10', '03:15', '03:20'], '1997-Jan-01'], [datetime.timedelta(seconds=40), ['03:00', '05', '10', '15', '20', '25', '30', '35', '40'], '1997-Jan-01 03:00'], [datetime.timedelta(seconds=2), ['59.5', '03:00', '00.5', '01.0', '01.5', '02.0', '02.5'], '1997-Jan-01 03:00'])
    new_tz = datetime.timezone(datetime.timedelta(hours=3))
    for t_delta, expected_strings, expected_offset in results:
        d2 = d1 + t_delta
        strings, offset = _create_auto_date_locator(d1, d2, new_tz)
        assert strings == expected_strings
        assert offset == expected_offset