import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_user_update_and_user_get_return_same_response(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    updated_user = {'enabled': False}
    updated_user_ref = PROVIDERS.identity_api.update_user(user['id'], updated_user)
    updated_user_ref.pop('extra', None)
    self.assertIs(False, updated_user_ref['enabled'])
    user_ref = PROVIDERS.identity_api.get_user(user['id'])
    self.assertDictEqual(updated_user_ref, user_ref)