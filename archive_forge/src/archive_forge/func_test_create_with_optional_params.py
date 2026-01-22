import uuid
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import endpoints
def test_create_with_optional_params(self):
    req_body = {'endpoint': {'region': 'RegionOne', 'publicurl': 'http://host-3:8774/v1.1/$(tenant_id)s', 'internalurl': 'http://host-3:8774/v1.1/$(tenant_id)s', 'adminurl': 'http://host-3:8774/v1.1/$(tenant_id)s', 'service_id': uuid.uuid4().hex}}
    resp_body = {'endpoint': {'adminurl': 'http://host-3:8774/v1.1/$(tenant_id)s', 'region': 'RegionOne', 'id': uuid.uuid4().hex, 'internalurl': 'http://host-3:8774/v1.1/$(tenant_id)s', 'publicurl': 'http://host-3:8774/v1.1/$(tenant_id)s'}}
    self.stub_url('POST', ['endpoints'], json=resp_body)
    endpoint = self.client.endpoints.create(region=req_body['endpoint']['region'], publicurl=req_body['endpoint']['publicurl'], adminurl=req_body['endpoint']['adminurl'], internalurl=req_body['endpoint']['internalurl'], service_id=req_body['endpoint']['service_id'])
    self.assertIsInstance(endpoint, endpoints.Endpoint)
    self.assertRequestBodyIs(json=req_body)