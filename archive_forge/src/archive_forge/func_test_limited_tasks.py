import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
def test_limited_tasks(self):
    """
        Ensure marker and limit query params work
        """
    path = '/v2/tasks'
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    tasks = json.loads(content)
    self.assertFalse(tasks['tasks'])
    task_ids = []
    task, _ = self._post_new_task(owner=TENANT1)
    task_ids.append(task['id'])
    task, _ = self._post_new_task(owner=TENANT2)
    task_ids.append(task['id'])
    task, _ = self._post_new_task(owner=TENANT3)
    task_ids.append(task['id'])
    path = '/v2/tasks'
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    tasks = json.loads(content)['tasks']
    self.assertEqual(3, len(tasks))
    params = 'limit=2'
    path = '/v2/tasks?%s' % params
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    actual_tasks = json.loads(content)['tasks']
    self.assertEqual(2, len(actual_tasks))
    self.assertEqual(tasks[0]['id'], actual_tasks[0]['id'])
    self.assertEqual(tasks[1]['id'], actual_tasks[1]['id'])
    params = 'marker=%s' % tasks[0]['id']
    path = '/v2/tasks?%s' % params
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    actual_tasks = json.loads(content)['tasks']
    self.assertEqual(2, len(actual_tasks))
    self.assertEqual(tasks[1]['id'], actual_tasks[0]['id'])
    self.assertEqual(tasks[2]['id'], actual_tasks[1]['id'])
    params = 'limit=1&marker=%s' % tasks[1]['id']
    path = '/v2/tasks?%s' % params
    response, content = self.http.request(path, 'GET', headers=minimal_task_headers())
    self.assertEqual(http.client.OK, response.status)
    actual_tasks = json.loads(content)['tasks']
    self.assertEqual(1, len(actual_tasks))
    self.assertEqual(tasks[2]['id'], actual_tasks[0]['id'])
    self._wait_on_task_execution()