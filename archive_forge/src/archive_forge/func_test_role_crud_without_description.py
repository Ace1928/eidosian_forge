import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_role_crud_without_description(self):
    role = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex, 'domain_id': None, 'options': {}}
    self.role_api.create_role(role['id'], role)
    role_ref = self.role_api.get_role(role['id'])
    role_ref_dict = {x: role_ref[x] for x in role_ref}
    self.assertIsNone(role_ref_dict['description'])
    role_ref_dict.pop('description')
    self.assertDictEqual(role, role_ref_dict)
    role['name'] = uuid.uuid4().hex
    updated_role_ref = self.role_api.update_role(role['id'], role)
    role_ref = self.role_api.get_role(role['id'])
    role_ref_dict = {x: role_ref[x] for x in role_ref}
    self.assertIsNone(updated_role_ref['description'])
    self.assertDictEqual(role_ref_dict, updated_role_ref)
    self.role_api.delete_role(role['id'])
    self.assertRaises(exception.RoleNotFound, self.role_api.get_role, role['id'])