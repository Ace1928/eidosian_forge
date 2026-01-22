import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_limit_with_invalid_input(self):
    ref = unit.new_limit_ref(project_id=self.project_id, service_id=self.service_id, region_id=self.region_id, resource_name='volume', resource_limit=10)
    r = self.post('/limits', body={'limits': [ref]}, token=self.system_admin_token, expected_status=http.client.CREATED)
    limit_id = r.result['limits'][0]['id']
    invalid_resource_limit_update = {'resource_limit': 'not_int'}
    invalid_description_update = {'description': 123}
    for input_limit in [invalid_resource_limit_update, invalid_description_update]:
        self.patch('/limits/%s' % limit_id, body={'limit': input_limit}, token=self.system_admin_token, expected_status=http.client.BAD_REQUEST)