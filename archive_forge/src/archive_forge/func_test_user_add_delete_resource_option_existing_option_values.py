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
def test_user_add_delete_resource_option_existing_option_values(self):
    user = self._create_user(self._get_user_dict())
    opt_value = uuid.uuid4().hex
    opt2_value = uuid.uuid4().hex
    user['options'][self.option1.option_name] = opt_value
    new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
    self.assertEqual(opt_value, new_ref['options'][self.option1.option_name])
    del user['options'][self.option1.option_name]
    user['options'][self.option2.option_name] = opt2_value
    new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
    self.assertEqual(opt_value, new_ref['options'][self.option1.option_name])
    self.assertEqual(opt2_value, new_ref['options'][self.option2.option_name])
    raw_ref = self._get_user_ref(user['id'])
    self.assertEqual(opt_value, raw_ref._resource_option_mapper[self.option1.option_id].option_value)
    self.assertEqual(opt2_value, raw_ref._resource_option_mapper[self.option2.option_id].option_value)
    user['options'][self.option1.option_name] = None
    new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
    self.assertNotIn(self.option1.option_name, new_ref['options'])
    self.assertEqual(opt2_value, new_ref['options'][self.option2.option_name])
    raw_ref = self._get_user_ref(user['id'])
    self.assertNotIn(raw_ref._resource_option_mapper, self.option1.option_id)
    self.assertEqual(opt2_value, raw_ref._resource_option_mapper[self.option2.option_id].option_value)