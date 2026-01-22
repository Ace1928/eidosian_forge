import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_duplicate_transition(self):
    m = self.jumper
    m.add_state('side_ways')
    self.assertRaises(excp.Duplicate, m.add_transition, 'up', 'side_ways', 'fall')