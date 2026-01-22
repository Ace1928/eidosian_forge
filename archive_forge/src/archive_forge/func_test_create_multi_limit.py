import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_multi_limit(self):
    ref1 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    ref2 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id2, resource_name='snapshot')
    r = self.post('/limits', body={'limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    limits = r.result['limits']
    for key in ['service_id', 'resource_name', 'resource_limit']:
        self.assertEqual(limits[0][key], ref1[key])
        self.assertEqual(limits[1][key], ref2[key])
    self.assertEqual(limits[0]['region_id'], ref1['region_id'])
    self.assertIsNone(limits[1].get('region_id'))