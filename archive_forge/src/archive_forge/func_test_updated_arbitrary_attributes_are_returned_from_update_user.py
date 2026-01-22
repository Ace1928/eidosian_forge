import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_updated_arbitrary_attributes_are_returned_from_update_user(self):
    attr_value = uuid.uuid4().hex
    user_data = unit.new_user_ref(domain_id=CONF.identity.default_domain_id, arbitrary_attr=attr_value)
    new_attr_value = uuid.uuid4().hex
    user = PROVIDERS.identity_api.create_user(user_data)
    user['arbitrary_attr'] = new_attr_value
    updated_user = PROVIDERS.identity_api.update_user(user['id'], user)
    self.assertEqual(new_attr_value, updated_user['arbitrary_attr'])