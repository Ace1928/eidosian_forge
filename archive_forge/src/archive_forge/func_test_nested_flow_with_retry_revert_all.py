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
def test_nested_flow_with_retry_revert_all(self):
    retry1 = retry.Times(0, 'r1', provides='x2', revert_all=True)
    flow = lf.Flow('flow-1').add(utils.ProgressingTask('task1'), lf.Flow('flow-2', retry1).add(utils.ConditionalTask('task2', inject={'x': 1})))
    engine = self._make_engine(flow)
    engine.storage.inject({'y': 2})
    with utils.CaptureListener(engine) as capturer:
        try:
            engine.run()
        except Exception:
            pass
    self.assertEqual({'y': 2}, engine.storage.fetch_all())
    expected = ['flow-1.f RUNNING', 'task1.t RUNNING', 'task1.t SUCCESS(5)', 'r1.r RUNNING', 'r1.r SUCCESS(1)', 'task2.t RUNNING', 'task2.t FAILURE(Failure: RuntimeError: Woot!)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'r1.r REVERTING', 'r1.r REVERTED(None)', 'task1.t REVERTING', 'task1.t REVERTED(None)', 'flow-1.f REVERTED']
    self.assertEqual(expected, capturer.values)