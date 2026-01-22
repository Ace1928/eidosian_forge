import uuid
import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_update_domain_name_or_id(self):
    domain_data = self._get_domain_data(description=self.getUniqueString('domainDesc'))
    domain_resource_uri = self.get_mock_url(append=[domain_data.domain_id])
    self.register_uris([dict(method='GET', uri=self.get_mock_url(append=[domain_data.domain_id]), status_code=200, json={'domain': domain_data.json_response['domain']}), dict(method='PATCH', uri=domain_resource_uri, status_code=200, json=domain_data.json_response, validate=dict(json=domain_data.json_request))])
    domain = self.cloud.update_domain(name_or_id=domain_data.domain_id, name=domain_data.domain_name, description=domain_data.description)
    self.assertThat(domain.id, matchers.Equals(domain_data.domain_id))
    self.assertThat(domain.name, matchers.Equals(domain_data.domain_name))
    self.assertThat(domain.description, matchers.Equals(domain_data.description))
    self.assert_calls()