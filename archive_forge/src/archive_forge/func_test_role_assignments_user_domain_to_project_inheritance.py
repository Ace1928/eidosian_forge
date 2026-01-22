from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_role_assignments_user_domain_to_project_inheritance(self):
    test_plan = {'entities': {'domains': {'users': 2, 'projects': 1}, 'roles': 3}, 'assignments': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'project': 0}, {'user': 0, 'role': 2, 'domain': 0, 'inherited_to_projects': True}, {'user': 1, 'role': 1, 'project': 0}], 'tests': [{'params': {'user': 0}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'project': 0}, {'user': 0, 'role': 2, 'domain': 0, 'inherited_to_projects': 'projects'}]}, {'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'project': 0}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'domain': 0}}]}, {'params': {'user': 0, 'project': 0, 'effective': True}, 'results': [{'user': 0, 'role': 1, 'project': 0}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'domain': 0}}]}]}
    self.execute_assignment_plan(test_plan)