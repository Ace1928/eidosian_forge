import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_registered_limit_with_invalid_input(self):
    ref1 = unit.new_registered_limit_ref()
    ref2 = unit.new_registered_limit_ref(default_limit='not_int')
    ref3 = unit.new_registered_limit_ref(resource_name=123)
    ref4 = unit.new_registered_limit_ref(region_id='fake_region')
    for input_limit in [ref1, ref2, ref3, ref4]:
        self.post('/registered_limits', body={'registered_limits': [input_limit]}, token=self.system_admin_token, expected_status=http.client.BAD_REQUEST)