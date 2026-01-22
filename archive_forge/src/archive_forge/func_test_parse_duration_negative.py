import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_negative(self):
    with self.assertRaises(ISOFormatError):
        parse_duration('P-1Y', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P-2M', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P-3D', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P-T4H', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P-T54M', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P-T6S', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P-7W', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P-1Y2M3DT4H54M6S', builder=None)