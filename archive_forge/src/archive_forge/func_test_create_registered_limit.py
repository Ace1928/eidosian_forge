import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_registered_limit(self):
    ref = unit.new_registered_limit_ref(service_id=self.service_id, region_id=self.region_id)
    r = self.post('/registered_limits', body={'registered_limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    registered_limits = r.result['registered_limits']
    for key in ['service_id', 'region_id', 'resource_name', 'default_limit', 'description']:
        self.assertEqual(registered_limits[0][key], ref[key])