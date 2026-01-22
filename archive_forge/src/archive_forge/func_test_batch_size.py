import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_batch_size(self):
    transport = oslo_messaging.get_notification_transport(self.conf, url='fake:')
    endpoint = mock.Mock()
    endpoint.info.return_value = None
    listener_thread = self._setup_listener(transport, [endpoint], batch=(5, None))
    notifier = self._setup_notifier(transport)
    ctxt = test_utils.TestContext()
    for _ in range(10):
        notifier.info(ctxt, 'an_event.start', 'test message')
    self.wait_for_messages(2)
    self.assertFalse(listener_thread.stop())
    messages = [dict(ctxt=ctxt, publisher_id='testpublisher', event_type='an_event.start', payload='test message', metadata={'message_id': mock.ANY, 'timestamp': mock.ANY})]
    endpoint.info.assert_has_calls([mock.call(messages * 5), mock.call(messages * 5)])