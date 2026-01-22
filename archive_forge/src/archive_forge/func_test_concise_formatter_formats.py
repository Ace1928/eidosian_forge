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
def test_concise_formatter_formats():
    formats = ['%Y', '%m/%Y', 'day: %d', '%H hr %M min', '%H hr %M min', '%S.%f sec']

    def _create_auto_date_locator(date1, date2):
        fig, ax = plt.subplots()
        locator = mdates.AutoDateLocator(interval_multiples=True)
        formatter = mdates.ConciseDateFormatter(locator, formats=formats)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(date1, date2)
        fig.canvas.draw()
        sts = [st.get_text() for st in ax.get_yticklabels()]
        return sts
    d1 = datetime.datetime(1997, 1, 1)
    results = ([datetime.timedelta(weeks=52 * 200), [str(t) for t in range(1980, 2201, 20)]], [datetime.timedelta(weeks=52), ['1997', '02/1997', '03/1997', '04/1997', '05/1997', '06/1997', '07/1997', '08/1997', '09/1997', '10/1997', '11/1997', '12/1997']], [datetime.timedelta(days=141), ['01/1997', 'day: 15', '02/1997', 'day: 15', '03/1997', 'day: 15', '04/1997', 'day: 15', '05/1997', 'day: 15']], [datetime.timedelta(days=40), ['01/1997', 'day: 05', 'day: 09', 'day: 13', 'day: 17', 'day: 21', 'day: 25', 'day: 29', '02/1997', 'day: 05', 'day: 09']], [datetime.timedelta(hours=40), ['day: 01', '04 hr 00 min', '08 hr 00 min', '12 hr 00 min', '16 hr 00 min', '20 hr 00 min', 'day: 02', '04 hr 00 min', '08 hr 00 min', '12 hr 00 min', '16 hr 00 min']], [datetime.timedelta(minutes=20), ['00 hr 00 min', '00 hr 05 min', '00 hr 10 min', '00 hr 15 min', '00 hr 20 min']], [datetime.timedelta(seconds=40), ['00 hr 00 min', '05.000000 sec', '10.000000 sec', '15.000000 sec', '20.000000 sec', '25.000000 sec', '30.000000 sec', '35.000000 sec', '40.000000 sec']], [datetime.timedelta(seconds=2), ['59.500000 sec', '00 hr 00 min', '00.500000 sec', '01.000000 sec', '01.500000 sec', '02.000000 sec', '02.500000 sec']])
    for t_delta, expected in results:
        d2 = d1 + t_delta
        strings = _create_auto_date_locator(d1, d2)
        assert strings == expected