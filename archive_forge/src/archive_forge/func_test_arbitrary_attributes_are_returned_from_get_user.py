import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_arbitrary_attributes_are_returned_from_get_user(self):
    attr_value = uuid.uuid4().hex
    user_data = unit.new_user_ref(domain_id=CONF.identity.default_domain_id, arbitrary_attr=attr_value)
    user_data = PROVIDERS.identity_api.create_user(user_data)
    user = PROVIDERS.identity_api.get_user(user_data['id'])
    self.assertEqual(attr_value, user['arbitrary_attr'])