import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_removed_kwarg_keeps_argspec(self):

    @removals.removed_kwarg('b')
    def f(b=2):
        return b

    def f_unwrapped(b=2):
        return b
    self.assertEqual(inspect.getfullargspec(f_unwrapped), inspect.getfullargspec(f))