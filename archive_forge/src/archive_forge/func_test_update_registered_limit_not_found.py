import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_update_registered_limit_not_found(self):
    update_ref = {'service_id': self.service_id, 'region_id': self.region_id, 'resource_name': 'snapshot', 'default_limit': 5}
    self.patch('/registered_limits/%s' % uuid.uuid4().hex, body={'registered_limit': update_ref}, token=self.system_admin_token, expected_status=http.client.NOT_FOUND)