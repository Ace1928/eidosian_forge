from saharaclient.api import job_binaries as jb
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_job_binary_get(self):
    url = self.URL + '/job-binaries/id'
    self.responses.get(url, json={'job_binary': self.body})
    resp = self.client.job_binaries.get('id')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, jb.JobBinaries)
    self.assertFields(self.body, resp)