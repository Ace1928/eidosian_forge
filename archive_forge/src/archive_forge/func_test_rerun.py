from oslo_serialization import jsonutils
from mistralclient.api.v2.executions import Execution
from mistralclient.api.v2 import tasks
from mistralclient.tests.unit.v2 import base
def test_rerun(self):
    url = self.TEST_URL + URL_TEMPLATE_ID % TASK['id']
    self.requests_mock.put(url, json=TASK)
    task = self.tasks.rerun(TASK['id'])
    self.assertDictEqual(tasks.Task(self.tasks, TASK).to_dict(), task.to_dict())
    body = {'reset': True, 'state': 'RUNNING', 'id': TASK['id']}
    self.assertDictEqual(body, self.requests_mock.last_request.json())