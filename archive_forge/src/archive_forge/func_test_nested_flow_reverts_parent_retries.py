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
def test_nested_flow_reverts_parent_retries(self):
    retry1 = retry.Times(3, 'r1', provides='x')
    retry2 = retry.Times(0, 'r2', provides='x2')
    flow = lf.Flow('flow-1', retry1).add(utils.ProgressingTask('task1'), lf.Flow('flow-2', retry2).add(utils.ConditionalTask('task2')))
    engine = self._make_engine(flow)
    engine.storage.inject({'y': 2})
    with utils.CaptureListener(engine) as capturer:
        engine.run()
    self.assertEqual({'y': 2, 'x': 2, 'x2': 1}, engine.storage.fetch_all())
    expected = ['flow-1.f RUNNING', 'r1.r RUNNING', 'r1.r SUCCESS(1)', 'task1.t RUNNING', 'task1.t SUCCESS(5)', 'r2.r RUNNING', 'r2.r SUCCESS(1)', 'task2.t RUNNING', 'task2.t FAILURE(Failure: RuntimeError: Woot!)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'r2.r REVERTING', 'r2.r REVERTED(None)', 'task1.t REVERTING', 'task1.t REVERTED(None)', 'r1.r RETRYING', 'task1.t PENDING', 'r2.r PENDING', 'task2.t PENDING', 'r1.r RUNNING', 'r1.r SUCCESS(2)', 'task1.t RUNNING', 'task1.t SUCCESS(5)', 'r2.r RUNNING', 'r2.r SUCCESS(1)', 'task2.t RUNNING', 'task2.t SUCCESS(None)', 'flow-1.f SUCCESS']
    self.assertEqual(expected, capturer.values)