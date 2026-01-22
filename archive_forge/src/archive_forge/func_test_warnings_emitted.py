import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_warnings_emitted(self):
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        self.assertEqual((2, 1), blip_blop(blip=2))
    self.assertEqual(1, len(capture))
    w = capture[0]
    self.assertEqual(DeprecationWarning, w.category)
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        self.assertEqual(2, blip_blop_3(blip=2))
    self.assertEqual(1, len(capture))
    w = capture[0]
    self.assertEqual(DeprecationWarning, w.category)