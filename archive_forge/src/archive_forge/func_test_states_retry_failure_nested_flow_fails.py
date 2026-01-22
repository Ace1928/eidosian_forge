import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import states as st
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.utils import eventlet_utils as eu
def test_states_retry_failure_nested_flow_fails(self):
    flow = lf.Flow('flow-1', utils.retry.AlwaysRevert('r1')).add(utils.TaskNoRequiresNoReturns('task1'), lf.Flow('flow-2', retry.Times(3, 'r2', provides='x')).add(utils.TaskNoRequiresNoReturns('task2'), utils.ConditionalTask('task3')), utils.TaskNoRequiresNoReturns('task4'))
    engine = self._make_engine(flow)
    engine.storage.inject({'y': 2})
    with utils.CaptureListener(engine) as capturer:
        engine.run()
    self.assertEqual({'y': 2, 'x': 2}, engine.storage.fetch_all())
    expected = ['flow-1.f RUNNING', 'r1.r RUNNING', 'r1.r SUCCESS(None)', 'task1.t RUNNING', 'task1.t SUCCESS(None)', 'r2.r RUNNING', 'r2.r SUCCESS(1)', 'task2.t RUNNING', 'task2.t SUCCESS(None)', 'task3.t RUNNING', 'task3.t FAILURE(Failure: RuntimeError: Woot!)', 'task3.t REVERTING', 'task3.t REVERTED(None)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'r2.r RETRYING', 'task2.t PENDING', 'task3.t PENDING', 'r2.r RUNNING', 'r2.r SUCCESS(2)', 'task2.t RUNNING', 'task2.t SUCCESS(None)', 'task3.t RUNNING', 'task3.t SUCCESS(None)', 'task4.t RUNNING', 'task4.t SUCCESS(None)', 'flow-1.f SUCCESS']
    self.assertEqual(expected, capturer.values)