import datetime
import uuid
import freezegun
import passlib.hash
from keystone.common import password_hashing
from keystone.common import provider_api
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import base
from keystone.identity.backends import resource_options as iro
from keystone.identity.backends import sql_model as model
from keystone.tests.unit import test_backend_sql
def test_user_set_option_in_resource_option(self):
    user = self._create_user(self._get_user_dict())
    opt_value = uuid.uuid4().hex
    user['options'][self.option1.option_name] = opt_value
    new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
    self.assertEqual(opt_value, new_ref['options'][self.option1.option_name])
    raw_ref = self._get_user_ref(user['id'])
    self.assertIn(self.option1.option_id, raw_ref._resource_option_mapper)
    self.assertEqual(opt_value, raw_ref._resource_option_mapper[self.option1.option_id].option_value)
    api_get_ref = PROVIDERS.identity_api.get_user(user['id'])
    self.assertEqual(opt_value, api_get_ref['options'][self.option1.option_name])