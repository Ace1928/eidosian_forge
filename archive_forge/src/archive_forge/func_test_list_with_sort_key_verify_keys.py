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
def test_list_with_sort_key_verify_keys(self):
    aliases = ['--sort_key', '--sort-key']
    for alias in aliases:
        for key in constants.SHARE_SORT_KEY_VALUES:
            for separator in self.separators:
                self.run_command('list ' + alias + separator + key)
                key = 'share_network_id' if key == 'share_network' else key
                key = 'snapshot_id' if key == 'snapshot' else key
                key = 'share_type_id' if key == 'share_type' else key
                key = 'availability_zone_id' if key == 'availability_zone' else key
                self.assert_called('GET', '/shares/detail?sort_key=' + key)