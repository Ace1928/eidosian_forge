import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_invalid_callbacks(self):
    m = self._create_fsm('working', add_states=['working', 'broken'])
    self.assertRaises(ValueError, m.add_state, 'b', on_enter=2)
    self.assertRaises(ValueError, m.add_state, 'b', on_exit=2)