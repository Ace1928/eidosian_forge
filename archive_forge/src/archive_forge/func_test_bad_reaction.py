import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_bad_reaction(self):
    m = self._create_fsm('unknown')
    self.assertRaises(excp.NotFound, m.add_reaction, 'something', 'boom', lambda *args: 'cough')