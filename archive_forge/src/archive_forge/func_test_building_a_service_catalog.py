import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_building_a_service_catalog(self):
    auth_ref = access.create(body=self.AUTH_RESPONSE_BODY)
    sc = auth_ref.service_catalog
    self.assertEqual(sc.url_for(service_type='compute'), 'https://compute.north.host/v1/1234')
    self.assertRaises(exceptions.EndpointNotFound, sc.url_for, region_name='South', service_type='compute')