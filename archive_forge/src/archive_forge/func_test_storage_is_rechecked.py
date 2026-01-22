import futurist
import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import linear_flow as lf
from taskflow import states
from taskflow import test
from taskflow.tests import utils
from taskflow.utils import eventlet_utils as eu
def test_storage_is_rechecked(self):
    flow = lf.Flow('linear').add(utils.ProgressingTask('b', requires=['foo']), utils.ProgressingTask('c'))
    engine = self._make_engine(flow)
    engine.storage.inject({'foo': 'bar'})
    with SuspendingListener(engine, task_name='b', task_state=states.SUCCESS):
        engine.run()
    self.assertEqual(states.SUSPENDED, engine.storage.get_flow_state())
    engine.storage.save(engine.storage.injector_name, {}, states.SUCCESS)
    self.assertRaises(exc.MissingDependencies, engine.run)