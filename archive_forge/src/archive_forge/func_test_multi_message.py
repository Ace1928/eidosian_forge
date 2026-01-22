import threading
from oslo_utils import uuidutils
from taskflow.engines.worker_based import dispatcher
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import proxy
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.types import latch
from taskflow.utils import threading_utils
def test_multi_message(self):
    message_count = 30
    barrier = latch.Latch(message_count)
    countdown = lambda data, message: barrier.countdown()
    on_notify = mock.MagicMock()
    on_notify.side_effect = countdown
    on_response = mock.MagicMock()
    on_response.side_effect = countdown
    on_request = mock.MagicMock()
    on_request.side_effect = countdown
    handlers = {pr.NOTIFY: dispatcher.Handler(on_notify), pr.RESPONSE: dispatcher.Handler(on_response), pr.REQUEST: dispatcher.Handler(on_request)}
    p = proxy.Proxy(TEST_TOPIC, TEST_EXCHANGE, handlers, transport='memory', transport_options={'polling_interval': POLLING_INTERVAL})
    t = threading_utils.daemon_thread(p.start)
    t.start()
    p.wait()
    for i in range(0, message_count):
        j = i % 3
        if j == 0:
            p.publish(pr.Notify(), TEST_TOPIC)
        elif j == 1:
            p.publish(pr.Response(pr.RUNNING), TEST_TOPIC)
        else:
            p.publish(pr.Request(test_utils.DummyTask('dummy_%s' % i), uuidutils.generate_uuid(), pr.EXECUTE, [], None), TEST_TOPIC)
    self.assertTrue(barrier.wait(test_utils.WAIT_TIMEOUT))
    self.assertEqual(0, barrier.needed)
    p.stop()
    t.join()
    self.assertTrue(on_notify.called)
    self.assertTrue(on_response.called)
    self.assertTrue(on_request.called)
    self.assertEqual(10, on_notify.call_count)
    self.assertEqual(10, on_response.call_count)
    self.assertEqual(10, on_request.call_count)
    call_count = sum([on_notify.call_count, on_response.call_count, on_request.call_count])
    self.assertEqual(message_count, call_count)