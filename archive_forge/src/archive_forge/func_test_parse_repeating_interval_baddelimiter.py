import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_parse_repeating_interval_baddelimiter(self):
    testtuples = ('R,PT1H2M,1980-03-05T01:01:00', 'R3 1981-04-05 P1D')
    for testtuple in testtuples:
        with self.assertRaises(ISOFormatError):
            parse_repeating_interval(testtuple, builder=None)