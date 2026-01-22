import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
@mock.patch('oslo_utils.eventletutils._eventlet')
def test_event_api_compat(self, mock_eventlet):
    with mock.patch('oslo_utils.eventletutils.is_monkey_patched', return_value=True):
        e_event = eventletutils.Event()
    self.assertIsInstance(e_event, eventletutils.EventletEvent)
    t_event = eventletutils.Event()
    t_event_cls = threading.Event
    self.assertIsInstance(t_event, t_event_cls)
    public_methods = [m for m in dir(t_event) if not m.startswith('_') and callable(getattr(t_event, m))]
    for method in public_methods:
        self.assertTrue(hasattr(e_event, method))
    e_event.set()
    self.assertTrue(e_event.isSet())
    e_event.set()
    self.assertTrue(e_event.isSet())