import threading
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging.rpc import dispatcher
from oslo_messaging.rpc import server as rpc_server_module
from oslo_messaging import server as server_module
from oslo_messaging.tests import utils as test_utils
def test_wait_for_running_task(self):
    start_event = eventletutils.Event()
    finish_event = eventletutils.Event()
    running_event = eventletutils.Event()
    done_event = eventletutils.Event()
    _runner = [None]

    class SteppingFakeExecutor(self.server._executor_cls):

        def __init__(self, *args, **kwargs):
            _runner[0] = eventlet.getcurrent()
            running_event.set()
            start_event.wait()
            super(SteppingFakeExecutor, self).__init__(*args, **kwargs)
            done_event.set()
            finish_event.wait()
    self.server._executor_cls = SteppingFakeExecutor
    start1 = eventlet.spawn(self.server.start)
    start2 = eventlet.spawn(self.server.start)
    running_event.wait()
    runner = _runner[0]
    waiter = start2 if runner == start1 else start2
    waiter_finished = eventletutils.Event()
    waiter.link(lambda _: waiter_finished.set())
    self.assertEqual(0, len(self.executors))
    self.assertFalse(waiter_finished.is_set())
    start_event.set()
    done_event.wait()
    self.assertEqual(1, len(self.executors))
    self.assertEqual([], self.executors[0]._calls)
    self.assertFalse(waiter_finished.is_set())
    finish_event.set()
    waiter.wait()
    runner.wait()
    self.assertTrue(waiter_finished.is_set())
    self.assertEqual(1, len(self.executors))
    self.assertEqual([], self.executors[0]._calls)