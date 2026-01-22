import uuid
import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_list_projects_search_compat(self):
    project_data = self._get_project_data(description=self.getUniqueString('projectDesc'))
    self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'projects': [project_data.json_response['project']]})])
    projects = self.cloud.search_projects(project_data.project_id)
    self.assertThat(len(projects), matchers.Equals(1))
    self.assertThat(projects[0].id, matchers.Equals(project_data.project_id))
    self.assert_calls()