from saharaclient.api import job_executions as je
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_job_executions_update(self):
    url = self.URL + '/job-executions/id'
    self.responses.patch(url, status_code=202, json=self.update_json)
    resp = self.client.job_executions.update('id', **self.update_json)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, je.JobExecution)
    self.assertEqual(self.update_json, json.loads(self.responses.last_request.body))
    self.client.job_executions.update('id')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual({}, json.loads(self.responses.last_request.body))
    unset_json = {'is_public': None, 'is_protected': None}
    self.client.job_executions.update('id', **unset_json)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(unset_json, json.loads(self.responses.last_request.body))