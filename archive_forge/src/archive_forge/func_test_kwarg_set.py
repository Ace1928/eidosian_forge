import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_kwarg_set(self):
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        self.assertEqual('The kitten meowed quietly', blip_blop_blip(type='kitten'))
    self.assertEqual(0, len(capture))