import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_create_then_set_parent(self):
    parent = self.resource_provider_create()
    wannabe_child = self.resource_provider_create()
    child = self.resource_provider_set(wannabe_child['uuid'], name='mandatory_name_1', parent_provider_uuid=parent['uuid'])
    self.assertEqual(child['parent_provider_uuid'], parent['uuid'])