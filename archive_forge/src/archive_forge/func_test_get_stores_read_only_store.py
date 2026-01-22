from oslo_config import cfg
import webob.exc
import glance.api.v2.discovery
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_get_stores_read_only_store(self):
    available_stores = ['cheap', 'fast', 'readonly_store', 'fast-cinder', 'fast-rbd', 'reliable']
    req = unit_test_utils.get_fake_request()
    output = self.controller.get_stores(req)
    self.assertIn('stores', output)
    for stores in output['stores']:
        self.assertIn('id', stores)
        self.assertIn(stores['id'], available_stores)
        if stores['id'] == 'readonly_store':
            self.assertTrue(stores['read-only'])
        else:
            self.assertIsNone(stores.get('read-only'))