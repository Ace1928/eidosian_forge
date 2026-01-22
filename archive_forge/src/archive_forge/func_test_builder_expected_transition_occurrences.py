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
def test_builder_expected_transition_occurrences(self):
    flow = lf.Flow('root')
    tasks = test_utils.make_many(10, task_cls=test_utils.TaskNoRequiresNoReturns)
    flow.add(*tasks)
    runtime, machine, memory, machine_runner = self._make_machine(flow, initial_state=st.RUNNING)
    transitions = list(machine_runner.run_iter(builder.START))
    occurrences = dict(((t, transitions.count(t)) for t in transitions))
    self.assertEqual(10, occurrences.get((st.SCHEDULING, st.WAITING)))
    self.assertEqual(10, occurrences.get((st.WAITING, st.ANALYZING)))
    self.assertEqual(9, occurrences.get((st.ANALYZING, st.SCHEDULING)))
    self.assertEqual(1, occurrences.get((builder.GAME_OVER, st.SUCCESS)))
    self.assertEqual(1, occurrences.get((builder.UNDEFINED, st.RESUMING)))
    self.assertEqual(0, len(memory.next_up))
    self.assertEqual(0, len(memory.not_done))
    self.assertEqual(0, len(memory.failures))