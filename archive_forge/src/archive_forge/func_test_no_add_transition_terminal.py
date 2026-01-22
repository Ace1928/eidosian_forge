import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_no_add_transition_terminal(self):
    m = self._create_fsm('up')
    m.add_state('down', terminal=True)
    self.assertRaises(excp.InvalidState, m.add_transition, 'down', 'up', 'jump')