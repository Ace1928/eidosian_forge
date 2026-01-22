from saharaclient.api import job_binaries as jb
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_create_job_binary(self):
    url = self.URL + '/job-binaries'
    self.responses.post(url, status_code=202, json={'job_binary': self.body})
    resp = self.client.job_binaries.create(**self.body)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(self.body, json.loads(self.responses.last_request.body))
    self.assertIsInstance(resp, jb.JobBinaries)
    self.assertFields(self.body, resp)