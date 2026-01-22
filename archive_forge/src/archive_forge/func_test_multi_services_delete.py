from unittest import mock
from unittest.mock import call
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import service
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
def test_multi_services_delete(self):
    arglist = []
    for s in self.services:
        arglist.append(s.binary)
    verifylist = [('service', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for s in self.services:
        calls.append(call(s.binary, ignore_missing=False))
    self.compute_sdk_client.delete_service.assert_has_calls(calls)
    self.assertIsNone(result)