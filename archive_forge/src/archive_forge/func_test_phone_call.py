import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_phone_call(self):
    handler = self._make_phone_call()
    r = runners.HierarchicalRunner(handler)
    r.run('call')
    self.assertTrue(handler.terminated)