import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
@mock.patch('oslo_utils.eventletutils._patcher')
def test_eventlet_is_patched(self, mock_patcher):
    mock_patcher.is_monkey_patched.return_value = True
    self.assertTrue(eventletutils.is_monkey_patched('os'))
    mock_patcher.is_monkey_patched.return_value = False
    self.assertFalse(eventletutils.is_monkey_patched('os'))