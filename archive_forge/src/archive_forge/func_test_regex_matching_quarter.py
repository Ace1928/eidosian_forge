from datetime import datetime
import numpy.testing as npt
from statsmodels.tsa.base.datetools import date_parser, dates_from_range
def test_regex_matching_quarter():
    t1 = '1999q4'
    t2 = '1999:q4'
    t3 = '1999:qIV'
    t4 = '1999qIV'
    result = datetime(1999, 12, 31)
    npt.assert_equal(date_parser(t1), result)
    npt.assert_equal(date_parser(t2), result)
    npt.assert_equal(date_parser(t3), result)
    npt.assert_equal(date_parser(t4), result)