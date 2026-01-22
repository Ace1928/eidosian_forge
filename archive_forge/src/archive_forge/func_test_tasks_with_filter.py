import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
def test_tasks_with_filter(self):
    path = '/v2/tasks'
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    content_dict = json.loads(content)
    self.assertFalse(content_dict['tasks'])
    task_ids = []
    task_owner = TENANT1
    data, req_input1 = self._post_new_task(owner=task_owner)
    task_ids.append(data['id'])
    task_owner = TENANT2
    data, req_input2 = self._post_new_task(owner=task_owner)
    task_ids.append(data['id'])
    path = '/v2/tasks'
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    content_dict = json.loads(content)
    self.assertEqual(2, len(content_dict['tasks']))
    params = 'owner=%s' % TENANT1
    path = '/v2/tasks?%s' % params
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    content_dict = json.loads(content)
    self.assertEqual(1, len(content_dict['tasks']))
    self.assertEqual(TENANT1, content_dict['tasks'][0]['owner'])
    params = 'owner=%s' % TENANT2
    path = '/v2/tasks?%s' % params
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    content_dict = json.loads(content)
    self.assertEqual(1, len(content_dict['tasks']))
    self.assertEqual(TENANT2, content_dict['tasks'][0]['owner'])
    params = 'type=import'
    path = '/v2/tasks?%s' % params
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    content_dict = json.loads(content)
    self.assertEqual(2, len(content_dict['tasks']))
    actual_task_ids = [task['id'] for task in content_dict['tasks']]
    self.assertEqual(set(task_ids), set(actual_task_ids))
    self._wait_on_task_execution()