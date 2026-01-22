from __future__ import absolute_import
import copy
import datetime
import pickle
import hypothesis
import hypothesis.extra.pytz
import hypothesis.strategies
import pytest
from . import iso8601
def test_fixedoffset_eq() -> None:
    expected_timezone = datetime.timezone(offset=datetime.timedelta(hours=2))
    assert expected_timezone == iso8601.FixedOffset(2, 0, '+2:00')