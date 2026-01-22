from saharaclient.api import data_sources as ds
from saharaclient.tests.unit import base
from unittest import mock
from oslo_serialization import jsonutils as json
def test_create_data_sources(self):
    url = self.URL + '/data-sources'
    self.responses.post(url, status_code=202, json={'data_source': self.response})
    resp = self.client.data_sources.create(**self.body)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(self.response, json.loads(self.responses.last_request.body))
    self.assertIsInstance(resp, ds.DataSources)
    self.assertFields(self.response, resp)