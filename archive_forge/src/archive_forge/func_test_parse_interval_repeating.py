import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_parse_interval_repeating(self):
    with self.assertRaises(ISOFormatError):
        parse_interval('R3/1981-04-05/P1D')
    with self.assertRaises(ISOFormatError):
        parse_interval('R3/1981-04-05/P0003-06-04T12:30:05.5')
    with self.assertRaises(ISOFormatError):
        parse_interval('R/PT1H2M/1980-03-05T01:01:00')