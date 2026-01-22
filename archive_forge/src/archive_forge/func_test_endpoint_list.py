from openstackclient.tests.functional.identity.v2 import common
def test_endpoint_list(self):
    endpoint_id = self._create_dummy_endpoint()
    raw_output = self.openstack('endpoint list')
    self.assertIn(endpoint_id, raw_output)
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, self.ENDPOINT_LIST_HEADERS)