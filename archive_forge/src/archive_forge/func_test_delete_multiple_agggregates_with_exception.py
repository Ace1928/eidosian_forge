from unittest import mock
from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import aggregate
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_delete_multiple_agggregates_with_exception(self):
    arglist = [self.fake_ags[0].id, 'unexist_aggregate']
    verifylist = [('aggregate', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.compute_sdk_client.find_aggregate.side_effect = [self.fake_ags[0], sdk_exceptions.NotFoundException]
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 aggregates failed to delete.', str(e))
    calls = []
    for a in arglist:
        calls.append(call(a, ignore_missing=False))
    self.compute_sdk_client.find_aggregate.assert_has_calls(calls)
    self.compute_sdk_client.delete_aggregate.assert_called_with(self.fake_ags[0].id, ignore_missing=False)