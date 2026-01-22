import math
import warnings
import matplotlib.dates
def mpl_dates_to_datestrings(dates, mpl_formatter):
    """Convert matplotlib dates to iso-formatted-like time strings.

    Plotly's accepted format: "YYYY-MM-DD HH:MM:SS" (e.g., 2001-01-01 00:00:00)

    Info on mpl dates: http://matplotlib.org/api/dates_api.html

    """
    _dates = dates
    if mpl_formatter == 'TimeSeries_DateFormatter':
        try:
            dates = matplotlib.dates.epoch2num([date * 24 * 60 * 60 for date in dates])
            dates = matplotlib.dates.num2date(dates)
        except:
            return _dates
    else:
        try:
            dates = matplotlib.dates.num2date(dates)
        except:
            return _dates
    time_stings = [' '.join(date.isoformat().split('+')[0].split('T')) for date in dates]
    return time_stings