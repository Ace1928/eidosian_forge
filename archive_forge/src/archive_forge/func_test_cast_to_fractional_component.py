import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_cast_to_fractional_component(self):
    self.assertEqual(_cast_to_fractional_component(10, '1.1'), FractionalComponent(1, 1))
    self.assertEqual(_cast_to_fractional_component(10, '-1.1'), FractionalComponent(-1, 1))
    self.assertEqual(_cast_to_fractional_component(100, '1.1'), FractionalComponent(1, 10))
    self.assertEqual(_cast_to_fractional_component(100, '-1.1'), FractionalComponent(-1, 10))