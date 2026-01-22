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
@ddt.unpack
@ddt.data({'expected_bool': True, 'snapshot_text': 'true', 'replication_type': 'readable'}, {'expected_bool': False, 'snapshot_text': 'false', 'replication_type': 'writable'})
def test_create_with_extra_specs(self, expected_bool, snapshot_text, replication_type):
    expected = {'share_type': {'name': 'test', 'share_type_access:is_public': True, 'extra_specs': {'driver_handles_share_servers': False, 'snapshot_support': expected_bool, 'replication_type': replication_type}}}
    self.run_command('type-create test false --extra-specs snapshot_support=' + snapshot_text + ' replication_type=' + replication_type)
    self.assert_called('POST', '/types', body=expected)