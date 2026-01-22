import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_get_duration_resolution_months(self):
    self.assertEqual(get_duration_resolution('P1Y2M'), DurationResolution.Months)
    self.assertEqual(get_duration_resolution('P1M'), DurationResolution.Months)
    self.assertEqual(get_duration_resolution('P1,5M'), DurationResolution.Months)
    self.assertEqual(get_duration_resolution('P1.5M'), DurationResolution.Months)