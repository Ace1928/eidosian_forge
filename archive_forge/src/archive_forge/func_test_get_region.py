import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_get_region(self):
    region = fixtures.Region(self.client)
    self.useFixture(region)
    region_ret = self.client.regions.get(region.id)
    self.check_region(region_ret, region.ref)