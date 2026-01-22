from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_delete_multiple_meters(self):
    arglist = []
    for n in self.meter_list:
        arglist.append(n.id)
    verifylist = [('meter', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for n in self.meter_list:
        calls.append(call(n))
    self.network_client.delete_metering_label.assert_has_calls(calls)
    self.assertIsNone(result)