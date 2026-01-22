from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_multi_subnet_pools_delete(self):
    arglist = []
    verifylist = []
    for s in self._subnet_pools:
        arglist.append(s.name)
    verifylist = [('subnet_pool', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for s in self._subnet_pools:
        calls.append(call(s))
    self.network_client.delete_subnet_pool.assert_has_calls(calls)
    self.assertIsNone(result)