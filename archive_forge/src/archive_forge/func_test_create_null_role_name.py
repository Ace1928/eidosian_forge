import uuid
from keystone.common import provider_api
from keystone.common import sql
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.assignment import test_core
from keystone.tests.unit.backend import core_sql
def test_create_null_role_name(self):
    role = unit.new_role_ref(name=None)
    self.assertRaises(exception.UnexpectedError, PROVIDERS.role_api.create_role, role['id'], role)
    self.assertRaises(exception.RoleNotFound, PROVIDERS.role_api.get_role, role['id'])