import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_datetime_to_secs():
    assert _helpers.datetime_to_secs(datetime.datetime(1970, 1, 1)) == 0
    assert _helpers.datetime_to_secs(datetime.datetime(1990, 5, 29)) == 643939200