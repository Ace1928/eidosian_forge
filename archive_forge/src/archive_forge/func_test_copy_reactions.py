import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_copy_reactions(self):
    c = self._create_fsm('down', add_start=False)
    d = c.copy()
    c.add_state('down')
    c.add_state('up')
    c.add_reaction('down', 'jump', lambda *args: 'up')
    c.add_transition('down', 'up', 'jump')
    self.assertEqual(1, c.events)
    self.assertEqual(0, d.events)
    self.assertNotIn('down', d)
    self.assertNotIn('up', d)
    self.assertEqual([], list(d))
    self.assertEqual([('down', 'jump', 'up')], list(c))