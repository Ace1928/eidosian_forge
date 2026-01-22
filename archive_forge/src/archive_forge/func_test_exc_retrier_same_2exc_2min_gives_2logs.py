import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
@mock.patch.object(logging, 'exception')
@mock.patch.object(timeutils, 'now')
def test_exc_retrier_same_2exc_2min_gives_2logs(self, mock_now, mock_log):
    self._exceptions = [Exception('unexpected 1'), Exception('unexpected 1')]
    mock_now.side_effect = [0, 65, 65, 66]
    self.exception_generator()
    self.assertEqual([], self._exceptions)
    self.assertEqual(4, len(mock_now.mock_calls))
    self.assertEqual(2, len(mock_log.mock_calls))
    mock_log.assert_has_calls([mock.call('Unexpected exception occurred 1 time(s)... retrying.'), mock.call('Unexpected exception occurred 1 time(s)... retrying.')])