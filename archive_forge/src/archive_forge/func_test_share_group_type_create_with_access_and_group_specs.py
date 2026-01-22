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
@ddt.data(True, False)
def test_share_group_type_create_with_access_and_group_specs(self, public):
    fake_share_type_1 = type('FakeShareType', (object,), {'id': '1234'})
    fake_share_type_2 = type('FakeShareType', (object,), {'id': '5678'})
    self.mock_object(shell_v2, '_find_share_type', mock.Mock(side_effect=[fake_share_type_1, fake_share_type_2]))
    expected = {'share_group_type': {'name': 'test-group-type-1', 'share_types': ['1234', '5678'], 'group_specs': {'spec1': 'value1'}, 'is_public': public}}
    self.run_command('share-group-type-create test-group-type-1 type1,type2 --is-public %s --group-specs spec1=value1' % str(public))
    self.assert_called_anytime('POST', '/share-group-types', body=expected)