import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_parse_interval_baddelimiter(self):
    testtuples = ('1980-03-05T01:01:00,1981-04-05T01:01:00', 'P1M 1981-04-05T01:01:00')
    for testtuple in testtuples:
        with self.assertRaises(ISOFormatError):
            parse_interval(testtuple, builder=None)