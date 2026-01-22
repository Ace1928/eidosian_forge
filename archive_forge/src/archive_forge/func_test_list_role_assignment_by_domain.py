from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_role_assignment_by_domain(self):
    """Test listing of role assignment filtered by domain."""
    test_plan = {'entities': {'domains': [{'users': 3, 'groups': 1}, 1], 'roles': 2}, 'group_memberships': [{'group': 0, 'users': [1, 2]}], 'assignments': [{'user': 0, 'role': 0, 'domain': 0}, {'group': 0, 'role': 1, 'domain': 0}], 'tests': [{'params': {'domain': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 1, 'role': 1, 'domain': 0, 'indirect': {'group': 0}}, {'user': 2, 'role': 1, 'domain': 0, 'indirect': {'group': 0}}]}, {'params': {'domain': 1, 'effective': True}, 'results': []}]}
    self.execute_assignment_plan(test_plan)