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
@ddt.data(('fake_type1',), ('fake_type1', 'fake_type2'))
def test_share_type_delete(self, type_ids):
    fake_share_types = [share_types.ShareType('fake', {'id': type_id}, True) for type_id in type_ids]
    self.mock_object(shell_v2, '_find_share_type', mock.Mock(side_effect=fake_share_types))
    self.run_command('type-delete %s' % ' '.join(type_ids))
    shell_v2._find_share_type.assert_has_calls([mock.call(self.shell.cs, t_id) for t_id in type_ids])
    for fake_share_type in fake_share_types:
        self.assert_called_anytime('DELETE', '/types/%s' % fake_share_type.id, clear_callstack=False)