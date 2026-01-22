import uuid
import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_create_project_v3(self):
    project_data = self._get_project_data(description=self.getUniqueString('projectDesc'), parent_id=uuid.uuid4().hex)
    reference_req = project_data.json_request.copy()
    reference_req['project']['enabled'] = True
    self.register_uris([dict(method='POST', uri=self.get_mock_url(), status_code=200, json=project_data.json_response, validate=dict(json=reference_req))])
    project = self.cloud.create_project(name=project_data.project_name, description=project_data.description, domain_id=project_data.domain_id, parent_id=project_data.parent_id)
    self.assertThat(project.id, matchers.Equals(project_data.project_id))
    self.assertThat(project.name, matchers.Equals(project_data.project_name))
    self.assertThat(project.description, matchers.Equals(project_data.description))
    self.assertThat(project.domain_id, matchers.Equals(project_data.domain_id))
    self.assert_calls()