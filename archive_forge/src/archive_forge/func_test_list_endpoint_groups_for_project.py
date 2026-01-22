import uuid
from keystoneclient.tests.unit.v3 import utils
def test_list_endpoint_groups_for_project(self):
    project_id = uuid.uuid4().hex
    endpoint_groups = {'endpoint_groups': [self.new_endpoint_group_ref(), self.new_endpoint_group_ref()]}
    self.stub_url('GET', [self.manager.OS_EP_FILTER_EXT, 'projects', project_id, 'endpoint_groups'], json=endpoint_groups, status_code=200)
    endpoint_groups_resp = self.manager.list_endpoint_groups_for_project(project=project_id)
    expected_endpoint_group_ids = [endpoint_group['id'] for endpoint_group in endpoint_groups['endpoint_groups']]
    actual_endpoint_group_ids = [endpoint_group.id for endpoint_group in endpoint_groups_resp]
    self.assertEqual(expected_endpoint_group_ids, actual_endpoint_group_ids)