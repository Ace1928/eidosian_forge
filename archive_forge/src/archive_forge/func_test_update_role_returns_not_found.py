import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_update_role_returns_not_found(self):
    role = unit.new_role_ref()
    self.assertRaises(exception.RoleNotFound, PROVIDERS.role_api.update_role, role['id'], role)