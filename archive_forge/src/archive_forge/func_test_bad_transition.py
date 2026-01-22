import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_bad_transition(self):
    m = self._create_fsm('unknown')
    m.add_state('fire')
    self.assertRaises(excp.NotFound, m.add_transition, 'unknown', 'something', 'boom')
    self.assertRaises(excp.NotFound, m.add_transition, 'something', 'unknown', 'boom')