import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_range_check_interval(self):
    with self.assertRaises(YearOutOfBoundsError):
        PythonTimeBuilder.build_interval(start=DateTuple('0007', None, None, None, None, None), duration=DurationTuple(None, None, None, str(datetime.timedelta.max.days), None, None, None))
    with self.assertRaises(YearOutOfBoundsError):
        PythonTimeBuilder.build_interval(start=DatetimeTuple(DateTuple('0007', None, None, None, None, None), TimeTuple('1', None, None, None)), duration=DurationTuple(str(datetime.timedelta.max.days // 365), None, None, None, None, None, None))
    with self.assertRaises(YearOutOfBoundsError):
        PythonTimeBuilder.build_interval(end=DateTuple('0001', None, None, None, None, None), duration=DurationTuple('3', None, None, None, None, None, None))
    with self.assertRaises(YearOutOfBoundsError):
        PythonTimeBuilder.build_interval(end=DatetimeTuple(DateTuple('0001', None, None, None, None, None), TimeTuple('1', None, None, None)), duration=DurationTuple('2', None, None, None, None, None, None))