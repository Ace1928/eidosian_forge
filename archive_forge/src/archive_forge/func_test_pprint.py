import pprint
import pytest
import pytz  # noqa  # a test below uses pytz but only inside a `eval` call
from pandas import Timestamp
def test_pprint(self):
    nested_obj = {'foo': 1, 'bar': [{'w': {'a': Timestamp('2011-01-01')}}] * 10}
    result = pprint.pformat(nested_obj, width=50)
    expected = "{'bar': [{'w': {'a': Timestamp('2011-01-01 00:00:00')}},\n         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},\n         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},\n         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},\n         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},\n         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},\n         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},\n         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},\n         {'w': {'a': Timestamp('2011-01-01 00:00:00')}},\n         {'w': {'a': Timestamp('2011-01-01 00:00:00')}}],\n 'foo': 1}"
    assert result == expected