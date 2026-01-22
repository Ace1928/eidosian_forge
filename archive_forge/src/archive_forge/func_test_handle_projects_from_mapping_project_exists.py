import uuid
from unittest import mock
from keystone.assignment.core import Manager as AssignmentApi
from keystone.auth.plugins import mapped
from keystone.exception import ProjectNotFound
from keystone.resource.core import Manager as ResourceApi
from keystone.tests import unit
def test_handle_projects_from_mapping_project_exists(self):
    project_mock_1 = self.create_project_mock_for_shadow_project(self.shadow_project_mock)
    project_mock_2 = self.create_project_mock_for_shadow_project(self.shadow_project_in_domain_mock)
    self.resource_api_mock.get_project_by_name.side_effect = [project_mock_1, project_mock_2]
    mapped.handle_projects_from_mapping(self.shadow_projects_mock, self.idp_domain_uuid_mock, self.existing_roles, self.user_mock, self.assignment_api_mock, self.resource_api_mock)
    self.resource_api_mock.get_project_by_name.assert_has_calls([mock.call(self.shadow_project_in_domain_mock['name'], self.shadow_project_in_domain_mock['domain']['id']), mock.call(self.shadow_project_mock['name'], self.idp_domain_uuid_mock)], any_order=True)
    self.assignment_api_mock.create_grant.assert_has_calls([mock.call(self.member_role_id, user_id=self.user_mock['id'], project_id=project_mock_1['id']), mock.call(self.member_role_id, user_id=self.user_mock['id'], project_id=project_mock_2['id'])])