import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_list_in_tree(self):
    rp1 = self.resource_provider_create()
    rp2 = self.resource_provider_create(parent_provider_uuid=rp1['uuid'])
    rp3 = self.resource_provider_create(parent_provider_uuid=rp1['uuid'])
    self.resource_provider_create()
    retrieved = self.resource_provider_list(in_tree=rp2['uuid'])
    self.assertEqual(set([rp['uuid'] for rp in retrieved]), set([rp1['uuid'], rp2['uuid'], rp3['uuid']]))