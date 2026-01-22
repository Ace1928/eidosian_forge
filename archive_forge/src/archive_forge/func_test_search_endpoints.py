import uuid
from testtools import matchers
from openstack.tests.unit import base
def test_search_endpoints(self):
    endpoints_data = [self._get_endpoint_v3_data(region='region1') for e in range(0, 2)]
    endpoints_data.extend([self._get_endpoint_v3_data() for e in range(1, 8)])
    self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'endpoints': [e.json_response['endpoint'] for e in endpoints_data]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'endpoints': [e.json_response['endpoint'] for e in endpoints_data]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'endpoints': [e.json_response['endpoint'] for e in endpoints_data]}), dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'endpoints': [e.json_response['endpoint'] for e in endpoints_data]})])
    endpoints = self.cloud.search_endpoints(id=endpoints_data[-1].endpoint_id)
    self.assertEqual(1, len(endpoints))
    self.assertThat(endpoints[0].id, matchers.Equals(endpoints_data[-1].endpoint_id))
    self.assertThat(endpoints[0].service_id, matchers.Equals(endpoints_data[-1].service_id))
    self.assertThat(endpoints[0].url, matchers.Equals(endpoints_data[-1].url))
    self.assertThat(endpoints[0].interface, matchers.Equals(endpoints_data[-1].interface))
    endpoints = self.cloud.search_endpoints(id='!invalid!')
    self.assertEqual(0, len(endpoints))
    endpoints = self.cloud.search_endpoints(filters={'region_id': 'region1'})
    self.assertEqual(2, len(endpoints))
    endpoints = self.cloud.search_endpoints(filters={'region_id': 'region1'})
    self.assertEqual(2, len(endpoints))
    self.assert_calls()