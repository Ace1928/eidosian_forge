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
def test_share_group_create_invalid_args(self):
    fake_share_type_1 = type('FakeShareType1', (object,), {'id': '1234'})
    fake_share_type_2 = type('FakeShareType2', (object,), {'id': '5678'})
    self.mock_object(shell_v2, '_find_share_type', mock.Mock(side_effect=[fake_share_type_1, fake_share_type_2]))
    fake_share_group_type = type('FakeShareGroupType', (object,), {'id': '2345'})
    self.mock_object(shell_v2, '_find_share_group_type', mock.Mock(return_value=fake_share_group_type))
    fake_share_group_snapshot = type('FakeShareGroupSnapshot', (object,), {'id': '3456'})
    self.mock_object(shell_v2, '_find_share_group_snapshot', mock.Mock(return_value=fake_share_group_snapshot))
    self.assertRaises(ValueError, self.run_command, 'share-group-create --name fake_sg --description my_group --share-types 1234,5678 --share-group-type fake_sg_type --source-share-group-snapshot fake_share_group_snapshot --availability-zone fake_az')