import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_get_hierarchy_as_ids(self):
    project = fixtures.Project(self.client, self.test_domain.id, parent=self.test_project.id)
    self.useFixture(project)
    child_project = fixtures.Project(self.client, self.test_domain.id, parent=project.id)
    self.useFixture(child_project)
    project_ret = self.client.projects.get(project.id, subtree_as_ids=True, parents_as_ids=True)
    self.assertCountEqual([self.test_project.id], project_ret.parents)
    self.assertCountEqual([child_project.id], project_ret.subtree)