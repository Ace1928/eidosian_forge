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
@ddt.data(({'key1': 'value1', 'key2': 'value2'}, {'key1': 'value1', 'key2': 'value2'}), ({'key1': {'key11': 'value11', 'key12': 'value12'}, 'key2': {'key21': 'value21'}}, {'key1': 'key11 = value11\nkey12 = value12', 'key2': 'key21 = value21'}), ({}, {}))
@ddt.unpack
@mock.patch.object(cliutils, 'print_dict', mock.Mock())
def test_quota_set_pretty_show(self, value, expected):
    fake_quota_set = fakes.FakeQuotaSet(value)
    shell_v2._quota_set_pretty_show(fake_quota_set)
    cliutils.print_dict.assert_called_with(expected)