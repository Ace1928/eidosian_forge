from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_route_overwrite_route(self):
    _testrouter = network_fakes.FakeRouter.create_one_router({'routes': [{'destination': '10.0.0.2', 'nexthop': '1.1.1.1'}]})
    self.network_client.find_router = mock.Mock(return_value=_testrouter)
    arglist = [_testrouter.name, '--route', 'destination=10.20.30.0/24,gateway=10.20.30.1', '--no-route']
    verifylist = [('router', _testrouter.name), ('routes', [{'destination': '10.20.30.0/24', 'gateway': '10.20.30.1'}]), ('no_route', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'routes': [{'destination': '10.20.30.0/24', 'nexthop': '10.20.30.1'}]}
    self.network_client.update_router.assert_called_once_with(_testrouter, **attrs)
    self.assertIsNone(result)