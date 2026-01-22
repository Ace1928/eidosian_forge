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
def test_run_iterations_suspended_failure(self):
    flow = lf.Flow('root')
    sad_tasks = test_utils.make_many(1, task_cls=test_utils.NastyFailingTask)
    flow.add(*sad_tasks)
    happy_tasks = test_utils.make_many(1, task_cls=test_utils.TaskNoRequiresNoReturns, offset=1)
    flow.add(*happy_tasks)
    runtime, machine, memory, machine_runner = self._make_machine(flow, initial_state=st.RUNNING)
    transitions = []
    for prior_state, new_state in machine_runner.run_iter(builder.START):
        transitions.append((new_state, memory.failures))
        if new_state == st.ANALYZING:
            runtime.storage.set_flow_state(st.SUSPENDED)
    state, failures = transitions[-1]
    self.assertEqual(st.SUSPENDED, state)
    self.assertEqual([], failures)
    self.assertEqual(st.PENDING, runtime.storage.get_atom_state(happy_tasks[0].name))
    self.assertEqual(st.FAILURE, runtime.storage.get_atom_state(sad_tasks[0].name))