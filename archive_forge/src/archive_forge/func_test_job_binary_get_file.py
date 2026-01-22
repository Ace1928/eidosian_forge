from saharaclient.api import job_binaries as jb
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_job_binary_get_file(self):
    url = self.URL + '/job-binaries/id/data'
    self.responses.get(url, text='data')
    resp = self.client.job_binaries.get_file('id')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(b'data', resp)