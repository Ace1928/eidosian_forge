from openstackclient.tests.functional.identity.v2 import common
def test_ec2_credentials_show(self):
    access_key = self._create_dummy_ec2_credentials()
    show_output = self.openstack('ec2 credentials show %s' % access_key)
    items = self.parse_show(show_output)
    self.assert_show_fields(items, self.EC2_CREDENTIALS_FIELDS)