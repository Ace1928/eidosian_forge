import itertools
from unittest import mock
import ddt
import fixtures
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient import client
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils as test_utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient import utils
from manilaclient.v2 import messages
from manilaclient.v2 import security_services
from manilaclient.v2 import share_access_rules
from manilaclient.v2 import share_group_types
from manilaclient.v2 import share_groups
from manilaclient.v2 import share_instances
from manilaclient.v2 import share_network_subnets
from manilaclient.v2 import share_networks
from manilaclient.v2 import share_servers
from manilaclient.v2 import share_snapshots
from manilaclient.v2 import share_types
from manilaclient.v2 import shares
from manilaclient.v2 import shell as shell_v2
@ddt.data(('share_xyz',), ('share_abc', 'share_xyz'))
def test_force_delete_wait(self, shares_to_delete):
    fake_manager = mock.Mock()
    fake_shares = [shares.Share(fake_manager, {'id': '1234'}) for share in shares_to_delete]
    share_not_found_error = "Delete for share %s failed: No share with a name or ID of '%s' exists."
    shares_are_not_found_errors = [exceptions.CommandError(share_not_found_error % (share, share)) for share in shares_to_delete]
    self.mock_object(shell_v2, '_find_share', mock.Mock(side_effect=fake_shares + shares_are_not_found_errors))
    self.run_command('force-delete %s --wait' % ' '.join(shares_to_delete))
    shell_v2._find_share.assert_has_calls([mock.call(self.shell.cs, share) for share in shares_to_delete])
    fake_manager.force_delete.assert_has_calls([mock.call(share) for share in fake_shares])
    self.assertEqual(len(shares_to_delete), fake_manager.force_delete.call_count)