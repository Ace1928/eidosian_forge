from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
def test_consistency_group_delete(self):
    arglist = [self.consistency_groups[0].id]
    verifylist = [('consistency_groups', [self.consistency_groups[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.consistencygroups_mock.delete.assert_called_with(self.consistency_groups[0].id, False)
    self.assertIsNone(result)