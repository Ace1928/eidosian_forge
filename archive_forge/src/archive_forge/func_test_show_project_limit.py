import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_show_project_limit(self):
    ref1 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume')
    ref2 = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id2, resource_name='snapshot')
    r = self.post('/limits', body={'limits': [ref1, ref2]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    if r.result['limits'][0]['resource_name'] == 'volume':
        id1 = r.result['limits'][0]['id']
    else:
        id1 = r.result['limits'][1]['id']
    self.get('/limits/fake_id', token=self.system_admin_token, expected_status=http.client.NOT_FOUND)
    r = self.get('/limits/%s' % id1, expected_status=http.client.OK)
    limit = r.result['limit']
    self.assertIsNone(limit['domain_id'])
    for key in ['service_id', 'region_id', 'resource_name', 'resource_limit', 'description', 'project_id']:
        self.assertEqual(limit[key], ref1[key])