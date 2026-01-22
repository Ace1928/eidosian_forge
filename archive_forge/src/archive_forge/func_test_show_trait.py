import uuid
from osc_placement.tests.functional import base
def test_show_trait(self):
    self.trait_create(TRAIT)
    self.assertEqual({'name': TRAIT}, self.trait_show(TRAIT))