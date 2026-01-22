import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_parse_repeating_interval_mockbuilder(self):
    mockBuilder = mock.Mock()
    args = {'R': False, 'Rnn': '3', 'interval': IntervalTuple(DateTuple('1981', '04', '05', None, None, None), None, DurationTuple(None, None, None, '1', None, None, None))}
    mockBuilder.build_repeating_interval.return_value = args
    result = parse_repeating_interval('R3/1981-04-05/P1D', builder=mockBuilder)
    self.assertEqual(result, args)
    mockBuilder.build_repeating_interval.assert_called_once_with(**args)
    mockBuilder = mock.Mock()
    args = {'R': False, 'Rnn': '11', 'interval': IntervalTuple(None, DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DurationTuple(None, None, None, None, '1', '2', None))}
    mockBuilder.build_repeating_interval.return_value = args
    result = parse_repeating_interval('R11/PT1H2M/1980-03-05T01:01:00', builder=mockBuilder)
    self.assertEqual(result, args)
    mockBuilder.build_repeating_interval.assert_called_once_with(**args)
    mockBuilder = mock.Mock()
    args = {'R': True, 'Rnn': None, 'interval': IntervalTuple(None, DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DurationTuple(None, None, None, None, '1', '2', None))}
    mockBuilder.build_repeating_interval.return_value = args
    result = parse_repeating_interval('R/PT1H2M/1980-03-05T01:01:00', builder=mockBuilder)
    self.assertEqual(result, args)
    mockBuilder.build_repeating_interval.assert_called_once_with(**args)