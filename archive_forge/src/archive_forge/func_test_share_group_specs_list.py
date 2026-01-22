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
@ddt.data(('', mock.ANY), (' --columns id,name', 'id,name'))
@ddt.unpack
def test_share_group_specs_list(self, args_cmd, expected_columns):
    self.mock_object(shell_v2, '_print_type_and_extra_specs_list')
    self.run_command('share-group-type-specs-list')
    self.assert_called('GET', '/share-group-types?is_public=all')
    shell_v2._print_type_and_extra_specs_list.assert_called_once_with(mock.ANY, columns=mock.ANY)