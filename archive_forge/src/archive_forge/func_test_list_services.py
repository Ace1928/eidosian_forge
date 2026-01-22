from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_list_services(self):
    service_data = self._get_service_data()
    self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'services': [service_data.json_response_v3['service']]})])
    services = self.cloud.list_services()
    self.assertThat(len(services), matchers.Equals(1))
    self.assertThat(services[0].id, matchers.Equals(service_data.service_id))
    self.assertThat(services[0].name, matchers.Equals(service_data.service_name))
    self.assertThat(services[0].type, matchers.Equals(service_data.service_type))
    self.assert_calls()