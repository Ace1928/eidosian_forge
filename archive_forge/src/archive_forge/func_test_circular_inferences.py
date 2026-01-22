from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_circular_inferences(self):
    """Test that implied roles are expanded out."""
    test_plan = {'entities': {'domains': {'users': 1, 'projects': 1}, 'roles': 4}, 'implied_roles': [{'role': 0, 'implied_roles': [1]}, {'role': 1, 'implied_roles': [2, 3]}, {'role': 3, 'implied_roles': [0]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}], 'tests': [{'params': {'user': 0}, 'results': [{'user': 0, 'role': 0, 'project': 0}]}, {'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 0, 'project': 0, 'indirect': {'role': 3}}, {'user': 0, 'role': 1, 'project': 0, 'indirect': {'role': 0}}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 3, 'project': 0, 'indirect': {'role': 1}}]}]}
    self.execute_assignment_plan(test_plan)