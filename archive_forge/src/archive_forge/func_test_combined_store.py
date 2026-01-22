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
def test_combined_store(self):
    components = self.make_components()
    components.conductor.connect()
    consumed_event = threading.Event()

    def on_consume(state, details):
        consumed_event.set()
    flow_store = {'x': True, 'y': False}
    job_store = {'z': None}
    components.board.notifier.register(base.REMOVAL, on_consume)
    with close_many(components.conductor, components.client):
        t = threading_utils.daemon_thread(components.conductor.run)
        t.start()
        lb, fd = pu.temporary_flow_detail(components.persistence, meta={'store': flow_store})
        engines.save_factory_details(fd, test_store_factory, [], {}, backend=components.persistence)
        components.board.post('poke', lb, details={'flow_uuid': fd.uuid, 'store': job_store})
        self.assertTrue(consumed_event.wait(test_utils.WAIT_TIMEOUT))
        components.conductor.stop()
        self.assertTrue(components.conductor.wait(test_utils.WAIT_TIMEOUT))
        self.assertFalse(components.conductor.dispatching)
    persistence = components.persistence
    with contextlib.closing(persistence.get_connection()) as conn:
        lb = conn.get_logbook(lb.uuid)
        fd = lb.find(fd.uuid)
    self.assertIsNotNone(fd)
    self.assertEqual(st.SUCCESS, fd.state)