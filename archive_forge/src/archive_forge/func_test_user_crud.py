import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_user_crud(self):
    user_dict = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    del user_dict['id']
    user = PROVIDERS.identity_api.create_user(user_dict)
    user_ref = PROVIDERS.identity_api.get_user(user['id'])
    del user_dict['password']
    user_ref_dict = {x: user_ref[x] for x in user_ref}
    self.assertLessEqual(user_dict.items(), user_ref_dict.items())
    user_dict['password'] = uuid.uuid4().hex
    PROVIDERS.identity_api.update_user(user['id'], user_dict)
    user_ref = PROVIDERS.identity_api.get_user(user['id'])
    del user_dict['password']
    user_ref_dict = {x: user_ref[x] for x in user_ref}
    self.assertLessEqual(user_dict.items(), user_ref_dict.items())
    PROVIDERS.identity_api.delete_user(user['id'])
    self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, user['id'])