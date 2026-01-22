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
@ddt.data({'driver_args': '--driver_options opt1=opt1 opt2=opt2', 'valid_params': {'driver_options': {'opt1': 'opt1', 'opt2': 'opt2'}}}, {'driver_args': '--driver_options opt1=opt1 opt2=opt2', 'subnet_id': 'fake_subnet_1', 'valid_params': {'driver_options': {'opt1': 'opt1', 'opt2': 'opt2'}}}, {'driver_args': '--driver_options opt1=opt1 opt2=opt2', 'valid_params': {'driver_options': {'opt1': 'opt1', 'opt2': 'opt2'}}, 'version': '2.51'}, {'driver_args': '--driver_options opt1=opt1 opt2=opt2', 'subnet_id': 'fake_subnet_1', 'valid_params': {'driver_options': {'opt1': 'opt1', 'opt2': 'opt2'}}, 'version': '2.51'}, {'driver_args': '', 'valid_params': {'driver_options': {}}, 'version': '2.51'}, {'driver_args': '--driver_options opt1=opt1 opt2=opt2', 'valid_params': {'driver_options': {'opt1': 'opt1', 'opt2': 'opt2'}}, 'version': '2.49'}, {'driver_args': '', 'valid_params': {'driver_options': {}}, 'network_id': 'fake_network_id', 'version': '2.49'}, {'driver_args': '', 'valid_params': {'driver_options': {}}, 'version': '2.49'})
@ddt.unpack
def test_share_server_manage_wait(self, driver_args, valid_params, version=None, network_id=None, subnet_id=None):
    fake_manager = mock.Mock()
    subnet_support = version is None or api_versions.APIVersion(version) >= api_versions.APIVersion('2.51')
    network_id = '3456' if network_id is None else network_id
    fake_share_network = share_networks.ShareNetwork(fake_manager, {'id': network_id, 'uuid': network_id})
    self.mock_object(shell_v2, '_find_share_network', mock.Mock(return_value=fake_share_network))
    fake_share_server = share_servers.ShareServer(fake_manager, {'id': 'fake'})
    self.mock_object(shell_v2, '_find_share_server', mock.Mock(return_value=fake_share_server))
    self.mock_object(shell_v2, '_wait_for_resource_status', mock.Mock())
    command = 'share-server-manage %(host)s %(share_network_id)s %(identifier)s %(driver_args)s ' % {'host': 'fake_host', 'share_network_id': fake_share_network.id, 'identifier': '88-as-23-f3-45', 'driver_args': driver_args}
    command += '--share-network-subnet %s' % subnet_id if subnet_id else ''
    self.run_command(command, version=version)
    expected = {'share_server': {'host': 'fake_host', 'share_network_id': fake_share_network.id, 'identifier': '88-as-23-f3-45', 'driver_options': driver_args}}
    if subnet_support:
        expected['share_server']['share_network_subnet_id'] = subnet_id
    expected['share_server'].update(valid_params)
    self.assert_called('POST', '/share-servers/manage', body=expected)
    shell_v2._wait_for_resource_status.assert_has_calls([mock.call(self.shell.cs, fake_share_server, resource_type='share_server', expected_status='active')])