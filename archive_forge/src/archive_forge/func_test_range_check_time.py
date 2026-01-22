import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_range_check_time(self):
    with self.assertRaises(MidnightBoundsError):
        PythonTimeBuilder.build_time(hh='24', mm='00', ss='01')
    with self.assertRaises(MidnightBoundsError):
        PythonTimeBuilder.build_time(hh='24', mm='00.1')
    with self.assertRaises(MidnightBoundsError):
        PythonTimeBuilder.build_time(hh='24', mm='01')
    with self.assertRaises(MidnightBoundsError):
        PythonTimeBuilder.build_time(hh='24.1')