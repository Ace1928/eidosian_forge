from openstackclient.tests.functional.identity.v2 import common
def test_ec2_credentials_multi_delete(self):
    access_key_1 = self._create_dummy_ec2_credentials(add_clean_up=False)
    access_key_2 = self._create_dummy_ec2_credentials(add_clean_up=False)
    raw_output = self.openstack('ec2 credentials delete ' + access_key_1 + ' ' + access_key_2)
    self.assertEqual(0, len(raw_output))