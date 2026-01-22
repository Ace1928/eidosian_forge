from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_list_projects_for_user_with_inherited_group_project_grants(self):
    """Test inherited role assignments for groups on nested projects.

        Test Plan:

        - Enable OS-INHERIT extension
        - Create a hierarchy of projects with one root and one leaf project
        - Assign an inherited group role on root project
        - Assign a non-inherited group role on root project
        - Get a list of projects for user, should return both projects
        - Disable OS-INHERIT extension
        - Get a list of projects for user, should return only root project

        """
    root_project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    root_project = PROVIDERS.resource_api.create_project(root_project['id'], root_project)
    leaf_project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id, parent_id=root_project['id'])
    leaf_project = PROVIDERS.resource_api.create_project(leaf_project['id'], leaf_project)
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
    group = PROVIDERS.identity_api.create_group(group)
    PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group['id'], project_id=root_project['id'], role_id=self.role_admin['id'], inherited_to_projects=True)
    PROVIDERS.assignment_api.create_grant(group_id=group['id'], project_id=root_project['id'], role_id=self.role_member['id'])
    user_projects = PROVIDERS.assignment_api.list_projects_for_user(user['id'])
    self.assertEqual(2, len(user_projects))
    self.assertIn(root_project, user_projects)
    self.assertIn(leaf_project, user_projects)
    test_plan = {'entities': {'domains': {'id': CONF.identity.default_domain_id, 'users': 1, 'groups': 1, 'projects': {'project': 1}}, 'roles': 2}, 'group_memberships': [{'group': 0, 'users': [0]}], 'assignments': [{'group': 0, 'role': 0, 'project': 0}, {'group': 0, 'role': 1, 'project': 0, 'inherited_to_projects': True}], 'tests': [{'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0, 'indirect': {'group': 0}}, {'user': 0, 'role': 1, 'project': 1, 'indirect': {'group': 0, 'project': 0}}]}]}
    self.execute_assignment_plan(test_plan)