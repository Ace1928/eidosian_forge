import uuid
import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_update_project_v3(self):
    project_data = self._get_project_data(description=self.getUniqueString('projectDesc'))
    reference_req = project_data.json_request.copy()
    reference_req['project'].pop('domain_id')
    reference_req['project'].pop('name')
    reference_req['project'].pop('enabled')
    self.register_uris([dict(method='GET', uri=self.get_mock_url(append=[project_data.project_id], qs_elements=['domain_id=' + project_data.domain_id]), status_code=200, json={'projects': [project_data.json_response['project']]}), dict(method='PATCH', uri=self.get_mock_url(append=[project_data.project_id]), status_code=200, json=project_data.json_response, validate=dict(json=reference_req))])
    project = self.cloud.update_project(project_data.project_id, description=project_data.description, domain_id=project_data.domain_id)
    self.assertThat(project.id, matchers.Equals(project_data.project_id))
    self.assertThat(project.name, matchers.Equals(project_data.project_name))
    self.assertThat(project.description, matchers.Equals(project_data.description))
    self.assert_calls()