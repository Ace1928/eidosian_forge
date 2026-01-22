import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_parse_datetime_badstr(self):
    testtuples = ('1981-04-05TA6:14:00.000123Z', '2004-W53-6T06:14:0B', '2014-01-230T23:21:28+00', '201401230T01:03:11.858714', '9999 W53T49', '9T0000,70:24,9', 'bad', '')
    for testtuple in testtuples:
        with self.assertRaises(ISOFormatError):
            parse_datetime(testtuple, builder=None)