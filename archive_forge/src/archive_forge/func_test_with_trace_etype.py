import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.stop')
@mock.patch('osprofiler.profiler.start')
def test_with_trace_etype(self, mock_start, mock_stop):

    def foo():
        with profiler.Trace('foo'):
            raise ValueError('bar')
    self.assertRaises(ValueError, foo)
    mock_start.assert_called_once_with('foo', info=None)
    mock_stop.assert_called_once_with(info={'etype': 'ValueError', 'message': 'bar'})