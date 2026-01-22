from oslo_config import cfg
import webob.exc
import glance.api.v2.discovery
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_get_stores_detail_with_store_weight(self):
    self.config(weight=100, group='fast')
    self.config(weight=200, group='cheap')
    self.config(weight=300, group='fast-rbd')
    self.config(weight=400, group='fast-cinder')
    self.config(weight=500, group='reliable')
    req = unit_test_utils.get_fake_request(roles=['admin'])
    output = self.controller.get_stores_detail(req)
    self.assertEqual(len(CONF.enabled_backends), len(output['stores']))
    self.assertIn('stores', output)
    for store in output['stores']:
        self.assertIn('weight', store)