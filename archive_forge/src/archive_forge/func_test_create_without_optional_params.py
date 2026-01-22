import uuid
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import endpoints
def test_create_without_optional_params(self):
    req_body_without_defaults = {'endpoint': {'region': 'RegionOne', 'service_id': uuid.uuid4().hex, 'publicurl': 'http://host-3:8774/v1.1/$(tenant_id)s', 'adminurl': None, 'internalurl': None}}
    resp_body = {'endpoint': {'region': 'RegionOne', 'id': uuid.uuid4().hex, 'publicurl': 'http://host-3:8774/v1.1/$(tenant_id)s', 'adminurl': None, 'internalurl': None}}
    self.stub_url('POST', ['endpoints'], json=resp_body)
    endpoint_without_defaults = self.client.endpoints.create(region=req_body_without_defaults['endpoint']['region'], publicurl=req_body_without_defaults['endpoint']['publicurl'], service_id=req_body_without_defaults['endpoint']['service_id'])
    self.assertIsInstance(endpoint_without_defaults, endpoints.Endpoint)
    self.assertRequestBodyIs(json=req_body_without_defaults)