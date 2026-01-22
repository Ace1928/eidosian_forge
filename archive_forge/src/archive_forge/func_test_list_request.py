import time
import uuid
from designateclient.tests import v2
def test_list_request(self):
    project = '098bee04-fe30-4a83-8ccd-e0c496755817'
    ref = [{'target_project_id': project}]
    parts = ['zones', 'tasks', 'transfer_requests']
    self.stub_url('GET', parts=parts, json={'transfer_requests': ref})
    self.client.zone_transfers.list_requests()
    self.assertRequestBodyIs('')