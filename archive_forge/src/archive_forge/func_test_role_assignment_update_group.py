import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import role_assignments
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
def test_role_assignment_update_group(self):
    prop_diff = {MixinClass.ROLES: [{'role': 'role_2', 'project': 'project_1'}, {'role': 'role_2', 'domain': 'domain_1'}]}
    self.test_role_assignment.update_assignment(group_id='group_1', prop_diff=prop_diff)
    self.roles.grant.assert_any_call(role='role_2', group='group_1', domain='domain_1')
    self.roles.grant.assert_any_call(role='role_2', group='group_1', project='project_1')
    self.roles.revoke.assert_any_call(role='role_1', group='group_1', domain='domain_1')
    self.roles.revoke.assert_any_call(role='role_1', group='group_1', project='project_1')