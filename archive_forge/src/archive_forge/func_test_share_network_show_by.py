from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data('name', 'id')
def test_share_network_show_by(self, attr):
    network_to_show = getattr(self.share_network, attr)
    fake_security_service = mock.Mock()
    fake_security_service.id = str(uuid.uuid4())
    fake_security_service.name = 'security-service-%s' % uuid.uuid4().hex
    self.security_services_mock.list = mock.Mock(return_value=[fake_security_service])
    arglist = [network_to_show]
    verifylist = [('share_network', network_to_show)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.find_resource', return_value=self.share_network) as find_resource:
        columns, data = self.cmd.take_action(parsed_args)
        find_resource.assert_called_once_with(self.share_networks_mock, network_to_show)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)