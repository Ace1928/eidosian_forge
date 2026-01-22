from openstackclient.tests.functional.identity.v2 import common
def test_ec2_credentials_list(self):
    self._create_dummy_ec2_credentials()
    raw_output = self.openstack('ec2 credentials list')
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, self.EC2_CREDENTIALS_LIST_HEADERS)