import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_parse_interval_end(self):
    self.assertEqual(_parse_interval_end('02', DateTuple('2020', '01', '01', None, None, None), 'T'), DateTuple(None, None, '02', None, None, None))
    self.assertEqual(_parse_interval_end('03-14', DateTuple('2008', '02', '15', None, None, None), 'T'), DateTuple(None, '03', '14', None, None, None))
    self.assertEqual(_parse_interval_end('0314', DateTuple('2008', '02', '15', None, None, None), 'T'), DateTuple(None, '03', '14', None, None, None))
    self.assertEqual(_parse_interval_end('15:30', DatetimeTuple(DateTuple('2007', '12', '14', None, None, None), TimeTuple('13', '30', None, None)), 'T'), TimeTuple('15', '30', None, None))
    self.assertEqual(_parse_interval_end('15T17:00', DatetimeTuple(DateTuple('2007', '11', '13', None, None, None), TimeTuple('09', '00', None, None)), 'T'), DatetimeTuple(DateTuple(None, None, '15', None, None, None), TimeTuple('17', '00', None, None)))
    self.assertEqual(_parse_interval_end('16T00:00', DatetimeTuple(DateTuple('2007', '11', '13', None, None, None), TimeTuple('00', '00', None, None)), 'T'), DatetimeTuple(DateTuple(None, None, '16', None, None, None), TimeTuple('00', '00', None, None)))
    self.assertEqual(_parse_interval_end('15 17:00', DatetimeTuple(DateTuple('2007', '11', '13', None, None, None), TimeTuple('09', '00', None, None)), ' '), DatetimeTuple(DateTuple(None, None, '15', None, None, None), TimeTuple('17', '00', None, None)))
    self.assertEqual(_parse_interval_end('12:34.567', DatetimeTuple(DateTuple('2007', '11', '13', None, None, None), TimeTuple('00', '00', None, None)), 'T'), TimeTuple('12', '34.567', None, None))