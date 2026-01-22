from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import ndp_proxy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_ndp_proxy_list_router(self):
    arglist = ['--router', 'fake-router-name']
    verifylist = [('router', 'fake-router-name')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.ndp_proxies.assert_called_once_with(**{'router_id': 'fake-router-id'})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))