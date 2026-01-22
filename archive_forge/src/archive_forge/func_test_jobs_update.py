from saharaclient.api import jobs
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_jobs_update(self):
    url = self.URL + '/jobs/id'
    update_body = {'name': 'new_name', 'description': 'description'}
    self.responses.patch(url, status_code=202, json=update_body)
    resp = self.client.jobs.update('id', name='new_name', description='description')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, jobs.Job)
    self.assertEqual(update_body, json.loads(self.responses.last_request.body))
    self.client.jobs.update('id')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual({}, json.loads(self.responses.last_request.body))
    unset_json = {'name': None, 'description': None, 'is_public': None, 'is_protected': None}
    self.client.jobs.update('id', **unset_json)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(unset_json, json.loads(self.responses.last_request.body))