import pickle
import unittest
from traits.api import HasTraits, Int, List, Map, on_trait_change, TraitError
def test_notification_init_value(self):
    preferences = Preferences(color='green')
    self.assertEqual(len(preferences.primary_changes), 1)
    self.assertEqual(len(preferences.shadow_changes), 1)