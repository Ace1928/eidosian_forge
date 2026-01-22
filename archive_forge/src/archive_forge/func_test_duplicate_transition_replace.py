import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_duplicate_transition_replace(self):
    m = self.jumper
    m.add_state('side_ways')
    m.add_transition('up', 'side_ways', 'fall', replace=True)