import datetime
from unittest import mock
import uuid
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import sql_model as model
from keystone.identity.shadow_backends import sql as shadow_sql
from keystone.tests import unit
def test_nonlocal_user_unique_user_id_constraint(self):
    user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.shadow_users_api.create_nonlocal_user(user_ref)
    nonlocal_user = {'domain_id': CONF.identity.default_domain_id, 'name': uuid.uuid4().hex, 'user_id': user['id']}
    self.assertRaises(sql.DBDuplicateEntry, self._add_nonlocal_user, nonlocal_user)