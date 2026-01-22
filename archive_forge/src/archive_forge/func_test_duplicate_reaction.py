import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_duplicate_reaction(self):
    self.assertRaises(excp.Duplicate, self.jumper.add_reaction, 'down', 'fall', lambda *args: 'skate')