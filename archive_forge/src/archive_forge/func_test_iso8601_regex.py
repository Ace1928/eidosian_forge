from __future__ import absolute_import
import copy
import datetime
import pickle
import hypothesis
import hypothesis.extra.pytz
import hypothesis.strategies
import pytest
from . import iso8601
def test_iso8601_regex() -> None:
    assert iso8601.ISO8601_REGEX.match('2006-10-11T00:14:33Z')