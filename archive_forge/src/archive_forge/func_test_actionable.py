import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_actionable(self):
    self.jumper.initialize()
    self.assertTrue(self.jumper.is_actionable_event('jump'))
    self.assertFalse(self.jumper.is_actionable_event('fall'))