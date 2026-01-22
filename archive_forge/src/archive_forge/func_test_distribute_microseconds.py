import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_distribute_microseconds(self):
    self.assertEqual(PythonTimeBuilder._distribute_microseconds(1, (), ()), (1,))
    self.assertEqual(PythonTimeBuilder._distribute_microseconds(11, (0,), (10,)), (1, 1))
    self.assertEqual(PythonTimeBuilder._distribute_microseconds(211, (0, 0), (100, 10)), (2, 1, 1))
    self.assertEqual(PythonTimeBuilder._distribute_microseconds(1, (), ()), (1,))
    self.assertEqual(PythonTimeBuilder._distribute_microseconds(11, (5,), (10,)), (6, 1))
    self.assertEqual(PythonTimeBuilder._distribute_microseconds(211, (10, 5), (100, 10)), (12, 6, 1))