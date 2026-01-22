import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
def test_task_schema_api(self):
    path = '/v2/schemas/task'
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    schema = tasks.get_task_schema()
    expected_schema = schema.minimal()
    data = json.loads(content)
    self.assertIsNotNone(data)
    self.assertEqual(expected_schema, data)
    path = '/v2/schemas/tasks'
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    schema = tasks.get_collection_schema()
    expected_schema = schema.minimal()
    data = json.loads(content)
    self.assertIsNotNone(data)
    self.assertEqual(expected_schema, data)
    self._wait_on_task_execution()