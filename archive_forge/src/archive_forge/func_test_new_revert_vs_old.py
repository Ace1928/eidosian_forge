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
def test_new_revert_vs_old(self):
    flow = lf.Flow('flow-1').add(utils.TaskNoRequiresNoReturns('task1'), lf.Flow('flow-2', retry.Times(1, 'r1', provides='x')).add(utils.TaskNoRequiresNoReturns('task2'), utils.ConditionalTask('task3')), utils.TaskNoRequiresNoReturns('task4'))
    engine = self._make_engine(flow)
    engine.storage.inject({'y': 2})
    with utils.CaptureListener(engine) as capturer:
        try:
            engine.run()
        except Exception:
            pass
    expected = ['flow-1.f RUNNING', 'task1.t RUNNING', 'task1.t SUCCESS(None)', 'r1.r RUNNING', 'r1.r SUCCESS(1)', 'task2.t RUNNING', 'task2.t SUCCESS(None)', 'task3.t RUNNING', 'task3.t FAILURE(Failure: RuntimeError: Woot!)', 'task3.t REVERTING', 'task3.t REVERTED(None)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'r1.r REVERTING', 'r1.r REVERTED(None)', 'flow-1.f REVERTED']
    self.assertEqual(expected, capturer.values)
    engine = self._make_engine(flow, defer_reverts=True)
    engine.storage.inject({'y': 2})
    with utils.CaptureListener(engine) as capturer:
        try:
            engine.run()
        except Exception:
            pass
    expected = ['flow-1.f RUNNING', 'task1.t RUNNING', 'task1.t SUCCESS(None)', 'r1.r RUNNING', 'r1.r SUCCESS(1)', 'task2.t RUNNING', 'task2.t SUCCESS(None)', 'task3.t RUNNING', 'task3.t FAILURE(Failure: RuntimeError: Woot!)', 'task3.t REVERTING', 'task3.t REVERTED(None)', 'task2.t REVERTING', 'task2.t REVERTED(None)', 'r1.r REVERTING', 'r1.r REVERTED(None)', 'task1.t REVERTING', 'task1.t REVERTED(None)', 'flow-1.f REVERTED']
    self.assertEqual(expected, capturer.values)