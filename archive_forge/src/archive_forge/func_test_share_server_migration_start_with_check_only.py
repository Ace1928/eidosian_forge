from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_server_migration_start_with_check_only(self):
    """Test share server migration start with check only"""
    arglist = ['1234', 'host@backend', '--preserve-snapshots', 'True', '--writable', 'True', '--nondisruptive', 'False', '--new-share-network', self.new_share_network.id, '--check-only']
    verifylist = [('share_server', '1234'), ('host', 'host@backend'), ('preserve_snapshots', 'True'), ('writable', 'True'), ('nondisruptive', 'False'), ('new_share_network', self.new_share_network.id), ('check_only', True)]
    expected_result = {'compatible': True, 'requested_capabilities': {'writable': 'True', 'nondisruptive': 'False', 'preserve_snapshots': 'True', 'share_network_id': None, 'host': 'host@backend'}, 'supported_capabilities': {'writable': True, 'nondisruptive': False, 'preserve_snapshots': True, 'share_network_id': self.new_share_network.id, 'migration_cancel': True, 'migration_get_progress': True}}
    self.share_server.migration_check.return_value = expected_result
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.57')
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.share_server.migration_check.assert_called_with('host@backend', 'True', 'False', 'True', self.new_share_network.id)
    result_dict = {}
    for count, column in enumerate(columns):
        result_dict[column] = data[count]
    self.assertEqual(expected_result, result_dict)