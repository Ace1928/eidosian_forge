from datetime import datetime
import numpy.testing as npt
from statsmodels.tsa.base.datetools import date_parser, dates_from_range
def test_dates_from_range():
    results = [datetime(1959, 3, 31, 0, 0), datetime(1959, 6, 30, 0, 0), datetime(1959, 9, 30, 0, 0), datetime(1959, 12, 31, 0, 0), datetime(1960, 3, 31, 0, 0), datetime(1960, 6, 30, 0, 0), datetime(1960, 9, 30, 0, 0), datetime(1960, 12, 31, 0, 0), datetime(1961, 3, 31, 0, 0), datetime(1961, 6, 30, 0, 0), datetime(1961, 9, 30, 0, 0), datetime(1961, 12, 31, 0, 0), datetime(1962, 3, 31, 0, 0), datetime(1962, 6, 30, 0, 0)]
    dt_range = dates_from_range('1959q1', '1962q2')
    npt.assert_(results == dt_range)
    results = results[2:]
    dt_range = dates_from_range('1959q3', length=len(results))
    npt.assert_(results == dt_range)
    results = [datetime(1959, 3, 31, 0, 0), datetime(1959, 4, 30, 0, 0), datetime(1959, 5, 31, 0, 0), datetime(1959, 6, 30, 0, 0), datetime(1959, 7, 31, 0, 0), datetime(1959, 8, 31, 0, 0), datetime(1959, 9, 30, 0, 0), datetime(1959, 10, 31, 0, 0), datetime(1959, 11, 30, 0, 0), datetime(1959, 12, 31, 0, 0), datetime(1960, 1, 31, 0, 0), datetime(1960, 2, 28, 0, 0), datetime(1960, 3, 31, 0, 0), datetime(1960, 4, 30, 0, 0), datetime(1960, 5, 31, 0, 0), datetime(1960, 6, 30, 0, 0), datetime(1960, 7, 31, 0, 0), datetime(1960, 8, 31, 0, 0), datetime(1960, 9, 30, 0, 0), datetime(1960, 10, 31, 0, 0), datetime(1960, 12, 31, 0, 0), datetime(1961, 1, 31, 0, 0), datetime(1961, 2, 28, 0, 0), datetime(1961, 3, 31, 0, 0), datetime(1961, 4, 30, 0, 0), datetime(1961, 5, 31, 0, 0), datetime(1961, 6, 30, 0, 0), datetime(1961, 7, 31, 0, 0), datetime(1961, 8, 31, 0, 0), datetime(1961, 9, 30, 0, 0), datetime(1961, 10, 31, 0, 0)]
    dt_range = dates_from_range('1959m3', length=len(results))