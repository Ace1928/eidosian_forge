import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_empty(self):
    self.AUTH_RESPONSE_BODY['access']['serviceCatalog'] = []
    auth_ref = access.create(body=self.AUTH_RESPONSE_BODY)
    self.assertRaises(exceptions.EmptyCatalog, auth_ref.service_catalog.url_for, service_type='image', interface='internalURL')