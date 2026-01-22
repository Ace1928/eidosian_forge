import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_build_terminal(self):
    space = [machines.State('down', is_terminal=False, next_states={'jump': 'fell_over'}), machines.State('fell_over', is_terminal=True)]
    m = machines.FiniteMachine.build(space)
    m.default_start_state = 'down'
    m.initialize()
    m.process_event('jump')
    self.assertTrue(m.terminated)