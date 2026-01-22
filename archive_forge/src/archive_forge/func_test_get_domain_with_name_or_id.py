import uuid
import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_get_domain_with_name_or_id(self):
    domain_data = self._get_domain_data()
    response = {'domains': [domain_data.json_response['domain']]}
    self.register_uris([dict(method='GET', uri=self.get_mock_url(append=[domain_data.domain_id]), status_code=200, json=domain_data.json_response), dict(method='GET', uri=self.get_mock_url(append=[domain_data.domain_name]), status_code=404), dict(method='GET', uri=self.get_mock_url(qs_elements=['name=' + domain_data.domain_name]), status_code=200, json=response)])
    domain = self.cloud.get_domain(name_or_id=domain_data.domain_id)
    domain_by_name = self.cloud.get_domain(name_or_id=domain_data.domain_name)
    self.assertThat(domain.id, matchers.Equals(domain_data.domain_id))
    self.assertThat(domain.name, matchers.Equals(domain_data.domain_name))
    self.assertThat(domain_by_name.id, matchers.Equals(domain_data.domain_id))
    self.assertThat(domain_by_name.name, matchers.Equals(domain_data.domain_name))
    self.assert_calls()