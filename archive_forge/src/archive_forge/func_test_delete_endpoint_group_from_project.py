import uuid
from keystoneclient.tests.unit.v3 import utils
def test_delete_endpoint_group_from_project(self):
    endpoint_group_id = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    self.stub_url('DELETE', [self.manager.OS_EP_FILTER_EXT, 'endpoint_groups', endpoint_group_id, 'projects', project_id], status_code=201)
    self.manager.delete_endpoint_group_from_project(project=project_id, endpoint_group=endpoint_group_id)