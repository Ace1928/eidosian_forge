import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_catalog(self):
    service_type = uuid.uuid4().hex
    service_name = uuid.uuid4().hex
    service_id = uuid.uuid4().hex
    region = uuid.uuid4().hex
    endpoints = {'public': uuid.uuid4().hex, 'internal': uuid.uuid4().hex, 'admin': uuid.uuid4().hex}
    token = fixture.V3Token()
    svc = token.add_service(type=service_type, name=service_name, id=service_id)
    svc.add_standard_endpoints(region=region, **endpoints)
    self.assertEqual(1, len(token['token']['catalog']))
    service = token['token']['catalog'][0]
    self.assertEqual(3, len(service['endpoints']))
    self.assertEqual(service_name, service['name'])
    self.assertEqual(service_type, service['type'])
    self.assertEqual(service_id, service['id'])
    for endpoint in service['endpoints']:
        self.assertTrue(endpoint.pop('id'))
    for interface, url in endpoints.items():
        endpoint = {'interface': interface, 'url': url, 'region': region, 'region_id': region}
        self.assertIn(endpoint, service['endpoints'])
    token.remove_service(type=service_type)
    self.assertEqual(0, len(token['token']['catalog']))