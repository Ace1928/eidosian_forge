from datetime import datetime
import pytest
from pandas._libs import tslib
from pandas import Timestamp
def test_parsers_iso8601_leading_space():
    date_str, expected = ('2013-1-1 5:30:00', datetime(2013, 1, 1, 5, 30))
    actual = tslib._test_parse_iso8601(' ' * 200 + date_str)
    assert actual == expected