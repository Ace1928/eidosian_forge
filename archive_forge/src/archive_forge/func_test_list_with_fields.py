from oslo_serialization import jsonutils
from mistralclient.api.v2.executions import Execution
from mistralclient.api.v2 import tasks
from mistralclient.tests.unit.v2 import base
def test_list_with_fields(self):
    field_params = '?fields=id,name'
    self.requests_mock.get(self.TEST_URL + URL_TEMPLATE + field_params, json={'tasks': [TASK]})
    self.tasks.list(fields=['id,name'])
    self.assertTrue(self.requests_mock.called_once)