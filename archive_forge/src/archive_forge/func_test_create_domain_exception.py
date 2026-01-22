import uuid
import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_create_domain_exception(self):
    domain_data = self._get_domain_data(domain_name='domain_name', enabled=True)
    with testtools.ExpectedException(exceptions.BadRequestException):
        self.register_uris([dict(method='POST', uri=self.get_mock_url(), status_code=400, json=domain_data.json_response, validate=dict(json=domain_data.json_request))])
        self.cloud.create_domain('domain_name')
    self.assert_calls()