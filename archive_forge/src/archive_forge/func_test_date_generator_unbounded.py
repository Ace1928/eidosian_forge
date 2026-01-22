import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_date_generator_unbounded(self):
    startdate = datetime.date(year=2018, month=8, day=29)
    timedelta = datetime.timedelta(days=5)
    generator = PythonTimeBuilder._date_generator_unbounded(startdate, timedelta)
    for dateindex in compat.range(0, 10):
        self.assertEqual(next(generator), datetime.date(year=2018, month=8, day=29) + dateindex * datetime.timedelta(days=5))