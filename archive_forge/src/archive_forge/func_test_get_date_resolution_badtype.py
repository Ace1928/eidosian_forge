import unittest
import aniso8601
from aniso8601.date import get_date_resolution, parse_date
from aniso8601.exceptions import DayOutOfBoundsError, ISOFormatError
from aniso8601.resolution import DateResolution
from aniso8601.tests.compat import mock
def test_get_date_resolution_badtype(self):
    testtuples = (None, 1, False, 1.234)
    for testtuple in testtuples:
        with self.assertRaises(ValueError):
            get_date_resolution(testtuple)