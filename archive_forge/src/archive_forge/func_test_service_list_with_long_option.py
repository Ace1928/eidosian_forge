from osc_lib.cli import format_columns
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backend
def test_service_list_with_long_option(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['Name', 'Capabilities']
    self.assertEqual(expected_columns, columns)
    datalist = ((self.pools.name, format_columns.DictColumn(self.pools.capabilities)),)
    self.assertEqual(datalist, tuple(data))
    self.volume_sdk_client.backend_pools.assert_called_with(detailed=True)