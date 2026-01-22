from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_server_adopt(self):
    arglist = ['somehost@backend', self.share_network['id'], 'share_server_identifier', '--share-network-subnet', self.share_network_subnet['id']]
    verifylist = [('host', 'somehost@backend'), ('share_network', self.share_network['id']), ('identifier', 'share_server_identifier'), ('share_network_subnet', self.share_network_subnet['id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.servers_mock.manage.assert_called_with(host='somehost@backend', share_network_id=self.share_network['id'], identifier='share_server_identifier', driver_options={}, share_network_subnet_id=self.share_network_subnet['id'])
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)