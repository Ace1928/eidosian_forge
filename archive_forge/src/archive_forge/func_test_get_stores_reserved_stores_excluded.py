from oslo_config import cfg
import webob.exc
import glance.api.v2.discovery
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_get_stores_reserved_stores_excluded(self):
    enabled_backends = {'fast': 'file', 'cheap': 'file'}
    self.config(enabled_backends=enabled_backends)
    req = unit_test_utils.get_fake_request()
    output = self.controller.get_stores(req)
    self.assertIn('stores', output)
    self.assertEqual(2, len(output['stores']))
    for stores in output['stores']:
        self.assertFalse(stores['id'].startswith('os_glance_'))