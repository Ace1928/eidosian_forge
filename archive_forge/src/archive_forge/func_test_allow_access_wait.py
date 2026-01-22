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
@ddt.data(*set(['2.45', api_versions.MAX_VERSION]))
def test_allow_access_wait(self, version):
    fake_access_rule = {'id': 'fake_id'}
    fake_access = mock.Mock()
    fake_access._info = fake_access_rule
    fake_share = mock.Mock()
    fake_share.name = 'fake_share'
    fake_share.allow = mock.Mock(return_value=fake_access_rule)
    self.mock_object(shell_v2, '_wait_for_resource_status', mock.Mock(return_value=fake_access))
    self.mock_object(share_access_rules.ShareAccessRuleManager, 'get', mock.Mock(return_value=fake_access_rule))
    with mock.patch.object(apiclient_utils, 'find_resource', mock.Mock(return_value=fake_share)):
        is_default_in_api = api_versions.APIVersion(version) >= api_versions.APIVersion('2.45')
        if is_default_in_api:
            self.run_command('access-allow fake_share ip 10.0.0.1 --wait', version=version)
            shell_v2._wait_for_resource_status.assert_has_calls([mock.call(self.shell.cs, fake_access_rule, resource_type='share_access_rule', expected_status='active', status_attr='state')])