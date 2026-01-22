import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_get_duration_resolution_minutes(self):
    self.assertEqual(get_duration_resolution('P1Y2M3DT4H5M'), DurationResolution.Minutes)
    self.assertEqual(get_duration_resolution('PT4H5M'), DurationResolution.Minutes)