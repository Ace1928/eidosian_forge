from oslo_config import cfg
import webob.exc
import glance.api.v2.discovery
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_get_stores_detail_properties(self):
    store_attributes = {'rbd': ['chunk_size', 'pool', 'thin_provisioning'], 'file': ['data_dir', 'chunk_size', 'thin_provisioning'], 'cinder': ['volume_type', 'use_multipath'], 'swift': ['container', 'large_object_size', 'large_object_chunk_size'], 'http': []}
    req = unit_test_utils.get_fake_request(roles=['admin'])
    output = self.controller.get_stores_detail(req)
    self.assertEqual(len(CONF.enabled_backends), len(output['stores']))
    self.assertIn('stores', output)
    for store in output['stores']:
        actual_attribute = list(store['properties'].keys())
        expected_attribute = store_attributes[store['type']]
        self.assertEqual(actual_attribute, expected_attribute)