from unittest import mock
import ddt
from manilaclient import base
from manilaclient.common import constants
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_servers
def test_migration_start(self):
    share_server = 'fake_share_server'
    host = 'fake_host'
    returned = 'fake'
    with mock.patch.object(self.manager, '_action', mock.Mock(return_value=returned)):
        result = self.manager.migration_start(share_server, host, writable=True, nondisruptive=True, preserve_snapshots=True)
        self.manager._action.assert_called_once_with('migration_start', share_server, {'host': host, 'writable': True, 'nondisruptive': True, 'preserve_snapshots': True, 'new_share_network_id': None})
        self.assertEqual(returned, result)