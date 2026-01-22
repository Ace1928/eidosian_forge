from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_role_assignments_unfiltered(self):
    """Test unfiltered listing of role assignments."""
    test_plan = {'entities': {'domains': {'users': 1, 'groups': 1, 'projects': 1}, 'roles': 3}, 'assignments': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'project': 0}, {'group': 0, 'role': 2, 'domain': 0}, {'group': 0, 'role': 2, 'project': 0}], 'tests': [{'params': {}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'project': 0}, {'group': 0, 'role': 2, 'domain': 0}, {'group': 0, 'role': 2, 'project': 0}]}]}
    self.execute_assignment_plan(test_plan)