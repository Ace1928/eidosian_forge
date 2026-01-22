import uuid
from keystoneclient.tests.unit.v3 import utils
def test_list_endpoints_for_project(self):
    project_id = uuid.uuid4().hex
    endpoints = {'endpoints': [self.new_endpoint_ref(), self.new_endpoint_ref()]}
    self.stub_url('GET', [self.manager.OS_EP_FILTER_EXT, 'projects', project_id, 'endpoints'], json=endpoints, status_code=200)
    endpoints_resp = self.manager.list_endpoints_for_project(project=project_id)
    expected_endpoint_ids = [endpoint['id'] for endpoint in endpoints['endpoints']]
    actual_endpoint_ids = [endpoint.id for endpoint in endpoints_resp]
    self.assertEqual(expected_endpoint_ids, actual_endpoint_ids)