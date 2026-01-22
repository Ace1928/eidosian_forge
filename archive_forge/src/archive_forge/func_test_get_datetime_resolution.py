import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_get_datetime_resolution(self):
    self.assertEqual(get_datetime_resolution('2019-06-05T01:03:11.858714'), TimeResolution.Seconds)
    self.assertEqual(get_datetime_resolution('2019-06-05T01:03:11'), TimeResolution.Seconds)
    self.assertEqual(get_datetime_resolution('2019-06-05T01:03'), TimeResolution.Minutes)
    self.assertEqual(get_datetime_resolution('2019-06-05T01'), TimeResolution.Hours)