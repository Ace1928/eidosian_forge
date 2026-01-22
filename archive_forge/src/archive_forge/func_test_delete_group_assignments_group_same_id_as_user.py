from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_delete_group_assignments_group_same_id_as_user(self):
    """Test deleting group assignments when group_id == user_id.

        In this scenario, only group assignments must be deleted (i.e.
        GROUP_DOMAIN or GROUP_PROJECT).

        Test plan:
        * Create a group and a user with the same ID;
        * Create four roles and assign them to both group and user;
        * Delete all group assignments;
        * User assignments must stay intact.
        """
    common_id = uuid.uuid4().hex
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project = PROVIDERS.resource_api.create_project(project['id'], project)
    user = unit.new_user_ref(id=common_id, domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.driver.create_user(common_id, user)
    self.assertEqual(common_id, user['id'])
    group = unit.new_group_ref(id=common_id, domain_id=CONF.identity.default_domain_id)
    group = PROVIDERS.identity_api.driver.create_group(common_id, group)
    self.assertEqual(common_id, group['id'])
    roles = []
    for _ in range(4):
        role = unit.new_role_ref()
        roles.append(PROVIDERS.role_api.create_role(role['id'], role))
    PROVIDERS.assignment_api.driver.create_grant(user_id=user['id'], domain_id=CONF.identity.default_domain_id, role_id=roles[0]['id'])
    PROVIDERS.assignment_api.driver.create_grant(user_id=user['id'], project_id=project['id'], role_id=roles[1]['id'])
    PROVIDERS.assignment_api.driver.create_grant(group_id=group['id'], domain_id=CONF.identity.default_domain_id, role_id=roles[2]['id'])
    PROVIDERS.assignment_api.driver.create_grant(group_id=group['id'], project_id=project['id'], role_id=roles[3]['id'])
    user_assignments = PROVIDERS.assignment_api.list_role_assignments(user_id=user['id'])
    self.assertThat(user_assignments, matchers.HasLength(2))
    group_assignments = PROVIDERS.assignment_api.list_role_assignments(group_id=group['id'])
    self.assertThat(group_assignments, matchers.HasLength(2))
    PROVIDERS.assignment_api.delete_group_assignments(group_id=group['id'])
    group_assignments = PROVIDERS.assignment_api.list_role_assignments(group_id=group['id'])
    self.assertThat(group_assignments, matchers.HasLength(0))
    user_assignments = PROVIDERS.assignment_api.list_role_assignments(user_id=user['id'])
    self.assertThat(user_assignments, matchers.HasLength(2))
    for assignment in group_assignments:
        self.assertThat(assignment.keys(), matchers.Contains('user_id'))