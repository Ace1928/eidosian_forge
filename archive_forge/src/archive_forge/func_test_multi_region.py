import itertools
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystonemiddleware.auth_token import _request
from keystonemiddleware.tests.unit import utils
def test_multi_region(self):
    token = fixture.V3Token()
    s = token.add_service(type='identity')
    s.add_endpoint('internal', self.INTERNAL_URL, region=self.REGION_ONE)
    s.add_endpoint('public', self.PUBLIC_URL, region=self.REGION_TWO)
    s.add_endpoint('admin', self.ADMIN_URL, region=self.REGION_THREE)
    auth_ref = access.create(body=token)
    catalog_data = auth_ref.service_catalog.catalog
    catalog = _request._normalize_catalog(catalog_data)
    self.assertEqual(1, len(catalog))
    service = catalog[0]
    expected = [{'internalURL': self.INTERNAL_URL, 'region': self.REGION_ONE}, {'publicURL': self.PUBLIC_URL, 'region': self.REGION_TWO}, {'adminURL': self.ADMIN_URL, 'region': self.REGION_THREE}]
    self.assertEqual('identity', service['type'])
    self.assertEqual(3, len(service['endpoints']))
    for e in expected:
        self.assertIn(e, expected)