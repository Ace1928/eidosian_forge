import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_list_regions(self):
    region_one = fixtures.Region(self.client)
    self.useFixture(region_one)
    region_two = fixtures.Region(self.client, parent_region=region_one.id)
    self.useFixture(region_two)
    regions = self.client.regions.list()
    for region in regions:
        self.check_region(region)
    self.assertIn(region_one.entity, regions)
    self.assertIn(region_two.entity, regions)