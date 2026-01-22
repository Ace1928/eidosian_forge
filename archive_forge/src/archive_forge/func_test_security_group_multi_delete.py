from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_multi_delete(self, sg_mock):
    sg_mock.return_value = mock.Mock(return_value=None)
    arglist = []
    verifylist = []
    for s in self._security_groups:
        arglist.append(s['id'])
    verifylist = [('group', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for s in self._security_groups:
        calls.append(call(s['id']))
    sg_mock.assert_has_calls(calls)
    self.assertIsNone(result)