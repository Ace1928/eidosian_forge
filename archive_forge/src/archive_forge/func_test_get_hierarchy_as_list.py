import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_get_hierarchy_as_list(self):
    project = fixtures.Project(self.client, self.test_domain.id, parent=self.test_project.id)
    self.useFixture(project)
    child_project = fixtures.Project(self.client, self.test_domain.id, parent=project.id)
    self.useFixture(child_project)
    role = fixtures.Role(self.client)
    self.useFixture(role)
    self.client.roles.grant(role.id, user=self.user_id, project=self.test_project.id)
    self.client.roles.grant(role.id, user=self.user_id, project=project.id)
    self.client.roles.grant(role.id, user=self.user_id, project=child_project.id)
    project_ret = self.client.projects.get(project.id, subtree_as_list=True, parents_as_list=True)
    self.check_project(project_ret, project.ref)
    self.assertCountEqual([{'project': self.test_project.entity.to_dict()}], project_ret.parents)
    self.assertCountEqual([{'project': child_project.entity.to_dict()}], project_ret.subtree)