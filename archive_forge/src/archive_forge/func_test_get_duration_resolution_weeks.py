import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_get_duration_resolution_weeks(self):
    self.assertEqual(get_duration_resolution('P1W'), DurationResolution.Weeks)
    self.assertEqual(get_duration_resolution('P1,5W'), DurationResolution.Weeks)
    self.assertEqual(get_duration_resolution('P1.5W'), DurationResolution.Weeks)