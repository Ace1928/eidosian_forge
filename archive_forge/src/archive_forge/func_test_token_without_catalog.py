import itertools
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystonemiddleware.auth_token import _request
from keystonemiddleware.tests.unit import utils
def test_token_without_catalog(self):
    token = fixture.V3Token()
    auth_ref = access.create(body=token)
    self.request.set_service_catalog_headers(auth_ref)
    self.assertNotIn('X-Service-Catalog', self.request.headers)