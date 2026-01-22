import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_limit_with_domain_as_project(self):
    ref = unit.new_limit_ref(project_id=self.domain_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    r = self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token)
    limits = r.result['limits']
    self.assertIsNone(limits[0]['project_id'])
    self.assertEqual(self.domain_id, limits[0]['domain_id'])