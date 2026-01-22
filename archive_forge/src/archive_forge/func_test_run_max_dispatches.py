import collections
import contextlib
import threading
import futurist
import testscenarios
from zake import fake_client
from taskflow.conductors import backends
from taskflow import engines
from taskflow.jobs.backends import impl_zookeeper
from taskflow.jobs import base
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import states as st
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as pu
from taskflow.utils import threading_utils
def test_run_max_dispatches(self):
    components = self.make_components()
    components.conductor.connect()
    consumed_event = threading.Event()

    def on_consume(state, details):
        consumed_event.set()
    components.board.notifier.register(base.REMOVAL, on_consume)
    with close_many(components.client, components.conductor):
        t = threading_utils.daemon_thread(lambda: components.conductor.run(max_dispatches=5))
        t.start()
        lb, fd = pu.temporary_flow_detail(components.persistence)
        engines.save_factory_details(fd, test_factory, [False], {}, backend=components.persistence)
        for _ in range(5):
            components.board.post('poke', lb, details={'flow_uuid': fd.uuid})
            self.assertTrue(consumed_event.wait(test_utils.WAIT_TIMEOUT))
        components.board.post('poke', lb, details={'flow_uuid': fd.uuid})
        components.conductor.stop()
        self.assertTrue(components.conductor.wait(test_utils.WAIT_TIMEOUT))
        self.assertFalse(components.conductor.dispatching)