import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
@unit.skip_if_cache_disabled('role')
def test_cache_layer_role_crud(self):
    role = unit.new_role_ref()
    role_id = role['id']
    PROVIDERS.role_api.create_role(role_id, role)
    role_ref = PROVIDERS.role_api.get_role(role_id)
    updated_role_ref = copy.deepcopy(role_ref)
    updated_role_ref['name'] = uuid.uuid4().hex
    PROVIDERS.role_api.driver.update_role(role_id, updated_role_ref)
    self.assertDictEqual(role_ref, PROVIDERS.role_api.get_role(role_id))
    PROVIDERS.role_api.get_role.invalidate(PROVIDERS.role_api, role_id)
    self.assertDictEqual(updated_role_ref, PROVIDERS.role_api.get_role(role_id))
    PROVIDERS.role_api.update_role(role_id, role_ref)
    self.assertDictEqual(role_ref, PROVIDERS.role_api.get_role(role_id))
    PROVIDERS.role_api.driver.delete_role(role_id)
    self.assertDictEqual(role_ref, PROVIDERS.role_api.get_role(role_id))
    PROVIDERS.role_api.get_role.invalidate(PROVIDERS.role_api, role_id)
    self.assertRaises(exception.RoleNotFound, PROVIDERS.role_api.get_role, role_id)
    PROVIDERS.role_api.create_role(role_id, role)
    PROVIDERS.role_api.get_role(role_id)
    PROVIDERS.role_api.delete_role(role_id)
    self.assertRaises(exception.RoleNotFound, PROVIDERS.role_api.get_role, role_id)