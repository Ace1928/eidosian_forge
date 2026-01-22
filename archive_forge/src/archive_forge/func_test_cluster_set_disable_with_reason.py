from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_cluster
def test_cluster_set_disable_with_reason(self):
    self.volume_client.api_version = api_versions.APIVersion('3.7')
    arglist = ['--binary', self.cluster.binary, '--disable', '--disable-reason', 'foo', self.cluster.name]
    verifylist = [('cluster', self.cluster.name), ('binary', self.cluster.binary), ('disabled', True), ('disabled_reason', 'foo')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, tuple(data))
    self.cluster_mock.update.assert_called_once_with(self.cluster.name, self.cluster.binary, disabled=True, disabled_reason='foo')