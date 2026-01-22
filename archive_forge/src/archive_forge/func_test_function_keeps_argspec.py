import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_function_keeps_argspec(self):
    self.assertEqual(inspect.getfullargspec(crimson_lightning_unwrapped), inspect.getfullargspec(crimson_lightning))