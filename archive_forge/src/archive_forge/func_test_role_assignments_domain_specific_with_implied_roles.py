from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_role_assignments_domain_specific_with_implied_roles(self):
    test_plan = {'entities': {'domains': {'users': 1, 'projects': 1, 'roles': 2}, 'roles': 2}, 'implied_roles': [{'role': 0, 'implied_roles': [1]}, {'role': 1, 'implied_roles': [2, 3]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}], 'tests': [{'params': {'user': 0}, 'results': [{'user': 0, 'role': 0, 'project': 0}]}, {'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 2, 'project': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 3, 'project': 0, 'indirect': {'role': 1}}]}]}
    self.execute_assignment_plan(test_plan)