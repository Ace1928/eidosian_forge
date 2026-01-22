import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
@mock.patch('oslo_utils.eventletutils._patcher')
def test_warning_not_patched_none_provided(self, mock_patcher):
    mock_patcher.already_patched = True
    mock_patcher.is_monkey_patched.return_value = False
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        eventletutils.warn_eventlet_not_patched()
    self.assertEqual(1, len(capture))
    w = capture[0]
    self.assertEqual(RuntimeWarning, w.category)
    for m in eventletutils._ALL_PATCH:
        self.assertIn(m, str(w.message))