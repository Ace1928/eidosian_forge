import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.stop')
@mock.patch('osprofiler.profiler.start')
def test_without_private(self, mock_start, mock_stop):
    fake_cls = FakeTraceWithMetaclassHideArgs()
    self.assertEqual(10, fake_cls._method(10))
    self.assertFalse(mock_start.called)
    self.assertFalse(mock_stop.called)