from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_cluster
def test_cluster_set_only_with_disable_reason(self):
    self.volume_client.api_version = api_versions.APIVersion('3.7')
    arglist = ['--disable-reason', 'foo', self.cluster.name]
    verifylist = [('cluster', self.cluster.name), ('binary', 'cinder-volume'), ('disabled', None), ('disabled_reason', 'foo')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('Cannot specify --disable-reason without --disable', str(exc))