from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_role_assignments_implied_roles_filtered_by_role(self):
    """Test that you can filter by role even if roles are implied."""
    test_plan = {'entities': {'domains': {'users': 1, 'projects': 2}, 'roles': 4}, 'implied_roles': [{'role': 0, 'implied_roles': 1}, {'role': 1, 'implied_roles': [2, 3]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 3, 'project': 1}], 'tests': [{'params': {'role': 3, 'effective': True}, 'results': [{'user': 0, 'role': 3, 'project': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 3, 'project': 1}]}]}
    self.execute_assignment_plan(test_plan)