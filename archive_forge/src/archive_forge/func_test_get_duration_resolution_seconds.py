import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_get_duration_resolution_seconds(self):
    self.assertEqual(get_duration_resolution('P1Y2M3DT4H54M6S'), DurationResolution.Seconds)
    self.assertEqual(get_duration_resolution('P1Y2M3DT4H54M6,5S'), DurationResolution.Seconds)
    self.assertEqual(get_duration_resolution('P1Y2M3DT4H54M6.5S'), DurationResolution.Seconds)
    self.assertEqual(get_duration_resolution('PT4H54M6,5S'), DurationResolution.Seconds)
    self.assertEqual(get_duration_resolution('PT4H54M6.5S'), DurationResolution.Seconds)
    self.assertEqual(get_duration_resolution('PT0.0000001S'), DurationResolution.Seconds)
    self.assertEqual(get_duration_resolution('PT2.0000048S'), DurationResolution.Seconds)
    self.assertEqual(get_duration_resolution('P0003-06-04T12:30:05'), DurationResolution.Seconds)
    self.assertEqual(get_duration_resolution('P0003-06-04T12:30:05.5'), DurationResolution.Seconds)
    self.assertEqual(get_duration_resolution('P0001-02-03T14:43:59.9999997'), DurationResolution.Seconds)