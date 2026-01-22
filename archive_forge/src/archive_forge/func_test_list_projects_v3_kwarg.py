import uuid
import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_list_projects_v3_kwarg(self):
    project_data = self._get_project_data(description=self.getUniqueString('projectDesc'))
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='projects?domain_id=%s' % project_data.domain_id), status_code=200, json={'projects': [project_data.json_response['project']]})])
    projects = self.cloud.list_projects(domain_id=project_data.domain_id)
    self.assertThat(len(projects), matchers.Equals(1))
    self.assertThat(projects[0].id, matchers.Equals(project_data.project_id))
    self.assert_calls()