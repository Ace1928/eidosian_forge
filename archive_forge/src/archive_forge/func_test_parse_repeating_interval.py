import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_parse_repeating_interval(self):
    with mock.patch.object(aniso8601.interval.PythonTimeBuilder, 'build_repeating_interval') as mockBuilder:
        expectedargs = {'R': False, 'Rnn': '3', 'interval': IntervalTuple(DateTuple('1981', '04', '05', None, None, None), None, DurationTuple(None, None, None, '1', None, None, None))}
        mockBuilder.return_value = expectedargs
        result = parse_repeating_interval('R3/1981-04-05/P1D')
        self.assertEqual(result, expectedargs)
        mockBuilder.assert_called_once_with(**expectedargs)
    with mock.patch.object(aniso8601.interval.PythonTimeBuilder, 'build_repeating_interval') as mockBuilder:
        expectedargs = {'R': False, 'Rnn': '11', 'interval': IntervalTuple(None, DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DurationTuple(None, None, None, None, '1', '2', None))}
        mockBuilder.return_value = expectedargs
        result = parse_repeating_interval('R11/PT1H2M/1980-03-05T01:01:00')
        self.assertEqual(result, expectedargs)
        mockBuilder.assert_called_once_with(**expectedargs)
    with mock.patch.object(aniso8601.interval.PythonTimeBuilder, 'build_repeating_interval') as mockBuilder:
        expectedargs = {'R': False, 'Rnn': '2', 'interval': IntervalTuple(DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DatetimeTuple(DateTuple('1981', '04', '05', None, None, None), TimeTuple('01', '01', '00', None)), None)}
        mockBuilder.return_value = expectedargs
        result = parse_repeating_interval('R2--1980-03-05T01:01:00--1981-04-05T01:01:00', intervaldelimiter='--')
        self.assertEqual(result, expectedargs)
        mockBuilder.assert_called_once_with(**expectedargs)
    with mock.patch.object(aniso8601.interval.PythonTimeBuilder, 'build_repeating_interval') as mockBuilder:
        expectedargs = {'R': False, 'Rnn': '2', 'interval': IntervalTuple(DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DatetimeTuple(DateTuple('1981', '04', '05', None, None, None), TimeTuple('01', '01', '00', None)), None)}
        mockBuilder.return_value = expectedargs
        result = parse_repeating_interval('R2/1980-03-05 01:01:00/1981-04-05 01:01:00', datetimedelimiter=' ')
        self.assertEqual(result, expectedargs)
        mockBuilder.assert_called_once_with(**expectedargs)
    with mock.patch.object(aniso8601.interval.PythonTimeBuilder, 'build_repeating_interval') as mockBuilder:
        expectedargs = {'R': True, 'Rnn': None, 'interval': IntervalTuple(None, DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), DurationTuple(None, None, None, None, '1', '2', None))}
        mockBuilder.return_value = expectedargs
        result = parse_repeating_interval('R/PT1H2M/1980-03-05T01:01:00')
        self.assertEqual(result, expectedargs)
        mockBuilder.assert_called_once_with(**expectedargs)