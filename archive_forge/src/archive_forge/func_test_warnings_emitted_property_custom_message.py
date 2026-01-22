import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
def test_warnings_emitted_property_custom_message(self):
    with warnings.catch_warnings(record=True) as capture:
        warnings.simplefilter('always')
        o = ThingB()
        self.assertEqual('green-blue', o.green_blue_tristars)
    self.assertEqual(1, len(capture))
    w = capture[0]
    self.assertIn('stop using me', str(w.message))
    self.assertEqual(DeprecationWarning, w.category)