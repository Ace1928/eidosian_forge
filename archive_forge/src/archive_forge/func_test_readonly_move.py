import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_readonly_move(self):
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        self.assertEqual('orange', Giraffe.colour)
        g = Giraffe()
        self.assertEqual(2, g.heightt)
    self.assertEqual(2, len(capture))