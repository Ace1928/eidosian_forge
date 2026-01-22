import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_range_check_date(self):
    with self.assertRaises(YearOutOfBoundsError):
        PythonTimeBuilder.build_date(YYYY='0000')
    with self.assertRaises(DayOutOfBoundsError):
        PythonTimeBuilder.build_date(YYYY='1981', DDD='366')