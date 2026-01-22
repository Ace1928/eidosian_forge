import uuid
from keystoneclient.tests.unit.v3 import utils
def test_list_projects_for_endpoint(self):
    endpoint_id = uuid.uuid4().hex
    projects = {'projects': [self.new_project_ref(), self.new_project_ref()]}
    self.stub_url('GET', [self.manager.OS_EP_FILTER_EXT, 'endpoints', endpoint_id, 'projects'], json=projects, status_code=200)
    projects_resp = self.manager.list_projects_for_endpoint(endpoint=endpoint_id)
    expected_project_ids = [project['id'] for project in projects['projects']]
    actual_project_ids = [project.id for project in projects_resp]
    self.assertEqual(expected_project_ids, actual_project_ids)