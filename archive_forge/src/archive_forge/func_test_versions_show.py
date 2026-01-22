from openstackclient.tests.functional import base
def test_versions_show(self):
    cmd_output = self.openstack('versions show', parse_output=True)
    self.assertIsNotNone(cmd_output)
    self.assertIn('Region Name', cmd_output[0])