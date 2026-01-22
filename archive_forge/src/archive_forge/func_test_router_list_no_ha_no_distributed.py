from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_router_list_no_ha_no_distributed(self):
    _routers = network_fakes.FakeRouter.create_routers({'ha': None, 'distributed': None}, count=3)
    arglist = []
    verifylist = [('long', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch.object(self.network_client, 'routers', return_value=_routers):
        columns, data = self.cmd.take_action(parsed_args)
    self.assertNotIn('is_distributed', columns)
    self.assertNotIn('is_ha', columns)