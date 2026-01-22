from osc_lib.cli import format_columns
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backend
def test_capability_show(self):
    arglist = ['fake']
    verifylist = [('host', 'fake')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['Title', 'Key', 'Type', 'Description']
    self.assertEqual(expected_columns, columns)
    capabilities = ['Compression', 'Replication', 'QoS', 'Thin Provisioning']
    for cap in data:
        self.assertIn(cap[0], capabilities)
    self.volume_sdk_client.get_capabilities.assert_called_with('fake')