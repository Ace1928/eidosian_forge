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
def test_role_assignment_delete_removed(self):
    self.test_role_assignment.parse_list_assignments.return_value = [{'role': 'role_1', 'domain': 'domain_1', 'project': None}]
    self.assertIsNone(self.test_role_assignment.delete_assignment(user_id='user_1'))
    expected = [({'role': 'role_1', 'user': 'user_1', 'domain': 'domain_1'},)]
    self.assertCountEqual(expected, self.roles.revoke.call_args_list)