from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_projects_for_user_with_inherited_grants(self):
    """Test inherited user roles.

        Test Plan:

        - Enable OS-INHERIT extension
        - Create a domain, with two projects and a user
        - Assign an inherited user role on the domain, as well as a direct
          user role to a separate project in a different domain
        - Get a list of projects for user, should return all three projects

        """
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user1 = unit.new_user_ref(domain_id=domain['id'])
    user1 = PROVIDERS.identity_api.create_user(user1)
    project1 = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    project2 = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project2['id'], project2)
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain['id'], role_id=self.role_admin['id'], inherited_to_projects=True)
    user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
    self.assertEqual(3, len(user_projects))
    test_plan = {'entities': {'domains': [{'projects': 1}, {'users': 1, 'projects': 2}], 'roles': 2}, 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 1, 'domain': 1, 'inherited_to_projects': True}], 'tests': [{'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 1, 'project': 1, 'indirect': {'domain': 1}}, {'user': 0, 'role': 1, 'project': 2, 'indirect': {'domain': 1}}]}]}
    self.execute_assignment_plan(test_plan)