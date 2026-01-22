import uuid
from osc_placement.tests.functional import base
def test_list_associated_traits(self):
    self.trait_create(TRAIT)
    rp = self.resource_provider_create()
    self.resource_provider_trait_set(rp['uuid'], TRAIT)
    self.assertIn(TRAIT, {t['name'] for t in self.trait_list(associated=True)})