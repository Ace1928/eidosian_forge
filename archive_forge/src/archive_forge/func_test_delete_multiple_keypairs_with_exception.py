import copy
from unittest import mock
from unittest.mock import call
import uuid
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import keypair
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_delete_multiple_keypairs_with_exception(self):
    arglist = [self.keypairs[0].name, 'unexist_keypair']
    verifylist = [('name', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.compute_sdk_client.delete_keypair.side_effect = [None, exceptions.CommandError]
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 keys failed to delete.', str(e))
    calls = []
    for k in arglist:
        calls.append(call(k, ignore_missing=False))
    self.compute_sdk_client.delete_keypair.assert_has_calls(calls)