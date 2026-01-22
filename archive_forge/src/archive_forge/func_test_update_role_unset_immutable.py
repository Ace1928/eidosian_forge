import copy
import uuid
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_update_role_unset_immutable(self):
    role = unit.new_role_ref()
    role_id = role['id']
    role['options'][ro_opt.IMMUTABLE_OPT.option_name] = True
    PROVIDERS.role_api.create_role(role_id, role)
    role_via_manager = PROVIDERS.role_api.get_role(role_id)
    self.assertTrue('options' in role_via_manager)
    self.assertTrue(role_via_manager['options'][ro_opt.IMMUTABLE_OPT.option_name])
    update_role = {'options': {ro_opt.IMMUTABLE_OPT.option_name: False}}
    PROVIDERS.role_api.update_role(role_id, update_role)
    role_via_manager = PROVIDERS.role_api.get_role(role_id)
    self.assertTrue('options' in role_via_manager)
    self.assertTrue(ro_opt.IMMUTABLE_OPT.option_name in role_via_manager['options'])
    self.assertFalse(role_via_manager['options'][ro_opt.IMMUTABLE_OPT.option_name])
    update_role = {'options': {ro_opt.IMMUTABLE_OPT.option_name: None}}
    role_updated = PROVIDERS.role_api.update_role(role_id, update_role)
    role_via_manager = PROVIDERS.role_api.get_role(role_id)
    self.assertTrue('options' in role_updated)
    self.assertTrue('options' in role_via_manager)
    self.assertFalse(ro_opt.IMMUTABLE_OPT.option_name in role_updated['options'])
    self.assertFalse(ro_opt.IMMUTABLE_OPT.option_name in role_via_manager['options'])