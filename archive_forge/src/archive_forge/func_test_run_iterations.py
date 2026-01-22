from automaton import exceptions as excp
from automaton import runners
from taskflow.engines.action_engine import builder
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import executor
from taskflow.engines.action_engine import runtime
from taskflow.patterns import linear_flow as lf
from taskflow import states as st
from taskflow import storage
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import notifier
from taskflow.utils import persistence_utils as pu
def test_run_iterations(self):
    flow = lf.Flow('root')
    tasks = test_utils.make_many(1, task_cls=test_utils.TaskNoRequiresNoReturns)
    flow.add(*tasks)
    runtime, machine, memory, machine_runner = self._make_machine(flow, initial_state=st.RUNNING)
    it = machine_runner.run_iter(builder.START)
    prior_state, new_state = next(it)
    self.assertEqual(st.RESUMING, new_state)
    self.assertEqual(0, len(memory.failures))
    prior_state, new_state = next(it)
    self.assertEqual(st.SCHEDULING, new_state)
    self.assertEqual(0, len(memory.failures))
    prior_state, new_state = next(it)
    self.assertEqual(st.WAITING, new_state)
    self.assertEqual(0, len(memory.failures))
    prior_state, new_state = next(it)
    self.assertEqual(st.ANALYZING, new_state)
    self.assertEqual(0, len(memory.failures))
    prior_state, new_state = next(it)
    self.assertEqual(builder.GAME_OVER, new_state)
    self.assertEqual(0, len(memory.failures))
    prior_state, new_state = next(it)
    self.assertEqual(st.SUCCESS, new_state)
    self.assertEqual(0, len(memory.failures))
    self.assertRaises(StopIteration, next, it)