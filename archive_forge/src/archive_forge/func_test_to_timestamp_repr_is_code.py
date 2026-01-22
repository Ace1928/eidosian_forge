import pprint
import pytest
import pytz  # noqa  # a test below uses pytz but only inside a `eval` call
from pandas import Timestamp
def test_to_timestamp_repr_is_code(self):
    zs = [Timestamp('99-04-17 00:00:00', tz='UTC'), Timestamp('2001-04-17 00:00:00', tz='UTC'), Timestamp('2001-04-17 00:00:00', tz='America/Los_Angeles'), Timestamp('2001-04-17 00:00:00', tz=None)]
    for z in zs:
        assert eval(repr(z)) == z