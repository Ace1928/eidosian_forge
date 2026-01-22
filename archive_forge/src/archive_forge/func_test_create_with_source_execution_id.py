from oslo_serialization import jsonutils
from mistralclient.api import base as api_base
from mistralclient.api.v2 import executions
from mistralclient.tests.unit.v2 import base
def test_create_with_source_execution_id(self):
    self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, json=SOURCE_EXEC, status_code=201)
    body = {'description': '', 'source_execution_id': SOURCE_EXEC['source_execution_id']}
    ex = self.executions.create(source_execution_id=SOURCE_EXEC['source_execution_id'])
    self.assertIsNotNone(ex)
    self.assertDictEqual(executions.Execution(self.executions, SOURCE_EXEC).to_dict(), ex.to_dict())
    self.assertDictEqual(body, self.requests_mock.last_request.json())