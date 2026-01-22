import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_build_transitions_with_callbacks(self):
    entered = collections.defaultdict(list)
    exitted = collections.defaultdict(list)

    def on_enter(state, event):
        entered[state].append(event)

    def on_exit(state, event):
        exitted[state].append(event)
    space = [machines.State('down', is_terminal=False, next_states={'jump': 'up'}, on_enter=on_enter, on_exit=on_exit), machines.State('up', is_terminal=False, next_states={'fall': 'down'}, on_enter=on_enter, on_exit=on_exit)]
    m = machines.FiniteMachine.build(space)
    m.default_start_state = 'down'
    expected = [('down', 'jump', 'up'), ('up', 'fall', 'down')]
    self.assertEqual(expected, list(m))
    m.initialize()
    m.process_event('jump')
    self.assertEqual({'down': ['jump']}, dict(exitted))
    self.assertEqual({'up': ['jump']}, dict(entered))
    m.process_event('fall')
    self.assertEqual({'down': ['jump'], 'up': ['fall']}, dict(exitted))
    self.assertEqual({'up': ['jump'], 'down': ['fall']}, dict(entered))