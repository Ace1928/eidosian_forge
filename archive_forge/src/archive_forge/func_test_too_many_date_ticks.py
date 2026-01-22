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
def test_too_many_date_ticks(caplog):
    caplog.set_level('WARNING')
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2000, 1, 20)
    fig, ax = plt.subplots()
    with pytest.warns(UserWarning) as rec:
        ax.set_xlim((t0, tf), auto=True)
        assert len(rec) == 1
        assert 'Attempting to set identical low and high xlims' in str(rec[0].message)
    ax.plot([], [])
    ax.xaxis.set_major_locator(mdates.DayLocator())
    v = ax.xaxis.get_major_locator()()
    assert len(v) > 1000
    assert caplog.records and all((record.name == 'matplotlib.ticker' and record.levelname == 'WARNING' for record in caplog.records))
    assert len(caplog.records) > 0