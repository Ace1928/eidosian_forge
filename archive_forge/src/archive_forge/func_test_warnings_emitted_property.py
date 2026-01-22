import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_warnings_emitted_property(self):
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        o = ThingB()
        self.assertEqual('green', o.green_tristars)
        o.green_tristars = 'b'
        del o.green_tristars
    self.assertEqual(3, len(capture))
    w = capture[0]
    self.assertEqual(DeprecationWarning, w.category)