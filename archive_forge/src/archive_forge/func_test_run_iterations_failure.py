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
def test_run_iterations_failure(self):
    flow = lf.Flow('root')
    tasks = test_utils.make_many(1, task_cls=test_utils.NastyFailingTask)
    flow.add(*tasks)
    runtime, machine, memory, machine_runner = self._make_machine(flow, initial_state=st.RUNNING)
    transitions = list(machine_runner.run_iter(builder.START))
    prior_state, new_state = transitions[-1]
    self.assertEqual(st.FAILURE, new_state)
    self.assertEqual(1, len(memory.failures))
    failure = memory.failures[0]
    self.assertTrue(failure.check(RuntimeError))
    self.assertEqual(st.REVERT_FAILURE, runtime.storage.get_atom_state(tasks[0].name))